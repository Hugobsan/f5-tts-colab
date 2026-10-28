"""TTS server: HTTP job queue served by GPU worker processes, engines loaded on demand.

Each job is a list of texts (one per narration paragraph) for one engine. The texts go
into that engine's queue and its workers pick them up in parallel, so a job finishes in
roughly (total / workers) time. Every request answers in milliseconds: the client
submits, polls and downloads, so nothing hits Cloudflare's 100 s limit.

Engines: every engine in F5_INSTALLED is installed, but only the default one starts
loaded (F5_WORKERS workers). A job for another engine makes the supervisor give it one
worker, retiring default-engine workers until the RAM budget fits (a retired worker
finishes the parts queued ahead of the retirement first). After F5_IDLE_UNLOAD seconds
without use the extra engine is unloaded and the default engine refills.

Workers load one at a time (loading peaks RAM) and a supervisor replaces any worker
that dies after becoming ready, putting its in-flight part back in the queue. A worker
that dies while loading is not retried: that is almost always lack of RAM, and it shows
up in /health with the exit code (for an on-demand engine, its pending jobs fail with it).

One more slot, "asr", runs faster-whisper (worker.asr_main) under the same supervisor: a job
sent with transcribe=true gets each finished part queued for it, and the text comes back in
the job status as heard{n: text}, so the client checks parts without its own Whisper.

Run:  F5_TOKEN=<secret> python run.py
Config (env):
  F5_TOKEN            required
  F5_INSTALLED        engines that may be requested, e.g. "f5,qwen3,chatterbox" (default: "f5")
  F5_DEFAULT_ENGINE   engine loaded at startup and used when a job names none (default: first installed)
  F5_WORKERS          workers of the default engine: "auto" (fit RAM/VRAM) or a number
  F5_IDLE_UNLOAD      seconds an on-demand engine stays loaded without use (default 600)
  F5_ENGINES          legacy "f5:auto,qwen3:1": names = F5_INSTALLED, first = default, its count = F5_WORKERS
  ENGINE_PYTHON_<n>   interpreter for engine <n> workers (its own venv); default: this Python
  F5_DEVICE           cuda | cpu (default: engine picks, cuda if available)
  F5_ASR_MODEL        faster-whisper model for the ASR slot (default "small"); "off" = no ASR
  F5_JOBS_DIR, PORT=8000; engine-specific vars live in engines/<name>.py
"""

import base64
import binascii
import multiprocessing as mp
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import engines
from worker import asr_main, worker_main

TOKEN = os.environ.get("F5_TOKEN", "")
_LEGACY = [s.strip() for s in os.environ.get("F5_ENGINES", "").split(",") if s.strip()]
INSTALLED = [n.strip() for n in (os.environ.get("F5_INSTALLED") or ",".join(s.partition(":")[0] for s in _LEGACY)
                                 or "f5").split(",") if n.strip()]
DEFAULT = os.environ.get("F5_DEFAULT_ENGINE") or INSTALLED[0]
WORKERS = os.environ.get("F5_WORKERS") or (_LEGACY[0].partition(":")[2] if _LEGACY else "") or "auto"
IDLE_UNLOAD = int(os.environ.get("F5_IDLE_UNLOAD", "600"))
ASR_MODEL = os.environ.get("F5_ASR_MODEL", "small")
ASR = "asr"  # slot "engine" name of the ASR worker
JOBS_DIR = Path(os.environ.get("F5_JOBS_DIR", "/content/jobs"))
PORT = int(os.environ.get("PORT", "8000"))
MAX_REF_BYTES = 10 * 1024 * 1024
# ponytail: per-worker costs are estimates (F5 measured ~2.3 GB RSS on a T4; engines declare RAM_MB);
# recalibrate from /health ram.workers_rss_mb and gpu.mem_used_mb if the supervisor packs badly.
DEFAULT_RAM_MB = 2500
VRAM_PER_WORKER_MB = 2000
ASR_RAM_MB = 1500
RESERVE_MB = 500  # kept free for the server itself (with 4 F5 + ASR a T4 session still had ~4.9 GB free)
MAX_AUTO_WORKERS = 8
MAX_RESTARTS = 5


class JobIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=2000)
    ref_audio: str = Field(description="reference wav, base64")
    ref_text: str = Field(min_length=1)
    engine: str | None = Field(None, description="engine id; default = F5_DEFAULT_ENGINE; others load on demand")
    nfe: int = Field(32, ge=1, le=64)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    opts: dict = Field(default_factory=dict, description="engine-specific options")
    transcribe: bool = Field(False, description="transcribe each finished part with the ASR worker (status heard{})")


# slot status: waiting -> loading -> ready (-> waiting again if it dies) | broken | retired
state = {"slots": {}, "next_wid": 0, "want": {}, "pills": {}, "last_used": {}, "engine_errors": {},
         "budget": 0, "n_default": 1, "ctx": None, "tasks": {}, "events": None, "stopping": False}
jobs: dict[str, dict] = {}
lock = threading.Lock()


def cost(engine):
    if engine == ASR:
        return ASR_RAM_MB
    return getattr(engines.get(engine), "RAM_MB", DEFAULT_RAM_MB)


def add_slot(engine):
    wid = state["next_wid"]
    state["next_wid"] += 1
    state["slots"][wid] = {"engine": engine, "proc": None, "status": "waiting", "device": None, "task": None,
                           "error": None, "restarts": 0, "last_death": None}
    return wid


def meminfo_mb(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) // 1024
    return 0


def rss_mb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def gpus():
    """One dict per GPU (Kaggle has 2 T4s, Colab 1); [] without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()
        rows = [[int(x) for x in line.split(", ")] for line in out]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return [{"util_pct": u, "mem_used_mb": used, "mem_total_mb": total} for u, used, total in rows]


def gpu_usage():
    """All GPUs summed (util averaged), for /health."""
    gs = gpus()
    if not gs:
        return None
    return {"count": len(gs), "util_pct": sum(g["util_pct"] for g in gs) // len(gs),
            "mem_used_mb": sum(g["mem_used_mb"] for g in gs), "mem_total_mb": sum(g["mem_total_mb"] for g in gs)}


def vram(engine):
    return getattr(engines.get(engine), "VRAM_MB", VRAM_PER_WORKER_MB)


def auto_workers(budget):
    """Default-engine workers that fit the RAM budget and the free VRAM of each GPU; 1 without a GPU."""
    gs = gpus()
    if not gs:
        return 1
    by_vram = sum((g["mem_total_mb"] - g["mem_used_mb"]) // vram(DEFAULT) for g in gs)
    return max(1, min(budget // cost(DEFAULT), by_vram, MAX_AUTO_WORKERS))


def pick_gpu(n_gpus):
    """GPU with the fewest TTS workers; the ASR worker stays on GPU 0."""
    load = [0] * n_gpus
    for s in state["slots"].values():
        if s.get("gpu") is not None and s["proc"] is not None and s["engine"] != ASR:
            load[s["gpu"]] += 1
    return load.index(min(load))


def exit_reason(code):
    if code == -signal.SIGKILL:
        return f"exit {code} (SIGKILL: quase sempre falta de RAM)"
    return f"exit {code}"


def asr_task(job_id, n):
    return job_id, n, str(JOBS_DIR / job_id / f"{n}.wav")


def asr_slot():
    return next((s for s in state["slots"].values() if s["engine"] == ASR), None)


def pending_parts(job):
    return job["total"] - len(job["done"] | job["failed"].keys())


def fit_want():
    """Default engine back to its size, then shrunk (and least recently used extras dropped) to fit the budget."""
    want = state["want"]
    want[DEFAULT] = state["n_default"]
    total = lambda: sum(cost(e) * n for e, n in want.items())  # noqa: E731
    while total() > state["budget"] and want[DEFAULT] > 1:
        want[DEFAULT] -= 1
    extras = sorted((e for e, n in want.items() if e != DEFAULT and n), key=lambda e: state["last_used"].get(e, 0))
    for e in extras[:-1]:  # never the most recent one: it is what was just asked for
        if total() <= state["budget"]:
            break
        want[e] = 0
        print(f"[tts] {e} descarregado para caber na RAM", flush=True)


def request_engine(engine):
    state["last_used"][engine] = time.time()
    if engine != DEFAULT and not state["want"].get(engine):
        state["want"][engine] = 1
        state["engine_errors"].pop(engine, None)
        fit_want()
        print(f"[tts] carregando {engine} sob demanda; {DEFAULT} fica com {state['want'][DEFAULT]} worker(s)", flush=True)


def engine_failed(engine, error):
    """An on-demand engine could not load: fail its pending parts, give the RAM back to the default engine."""
    state["want"][engine] = 0
    state["pills"][engine] = 0
    state["engine_errors"][engine] = error
    for j in jobs.values():
        if j["engine"] == engine:
            for n in range(j["total"]):
                if n not in j["done"] and n not in j["failed"]:
                    j["failed"][n] = f"motor {engine} não carregou: {error}"
    for wid in [w for w, s in state["slots"].items() if s["engine"] == engine and s["proc"] is None]:
        del state["slots"][wid]
    fit_want()
    print(f"[tts] motor {engine} falhou ({error}); {DEFAULT} volta a {state['want'][DEFAULT]} worker(s)", flush=True)


def requeue(slot, reason):
    """Puts the part a dead worker was generating (or transcribing) back in its queue, once."""
    task, slot["task"] = slot["task"], None
    if task is None:
        return
    job_id, n = task
    job = jobs.get(job_id)
    asr = slot["engine"] == ASR
    if job is None or n in (job["heard"] if asr else job["done"]):
        return
    if not asr:
        job["running"].discard(n)
    retries = job["asr_retries" if asr else "retries"]
    retries[n] = retries.get(n, 0) + 1
    if retries[n] > 1:
        job["failed"][n] = f"worker {'do ASR ' if asr else ''}morreu duas vezes nesta parte ({reason})"
    elif asr:
        state["tasks"][ASR].put(asr_task(job_id, n))
    else:
        state["tasks"][job["engine"]].put(job["tasks"][n])


def start_worker(wid, s):
    asr = s["engine"] == ASR
    # set_executable is global to spawn; only this supervisor thread starts processes, so it is safe here.
    # The ASR worker runs in the server's own env (faster-whisper is in server/requirements.txt).
    state["ctx"].set_executable(sys.executable if asr else os.environ.get(f"ENGINE_PYTHON_{s['engine']}", sys.executable))
    s["proc"] = state["ctx"].Process(
        target=asr_main if asr else worker_main,
        args=(wid, ASR_MODEL if asr else s["engine"], state["tasks"][s["engine"]], state["events"]), daemon=True)
    n_gpus = len(gpus())
    s["gpu"] = None
    if n_gpus > 1:  # spawned children inherit os.environ, so each sees only its own GPU (as cuda:0)
        s["gpu"] = 0 if asr else pick_gpu(n_gpus)
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(s["gpu"])
    try:
        s["proc"].start()
    finally:
        if s["gpu"] is not None:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old
    s["status"] = "loading"


def reap():
    """Handles workers whose process is gone: retired ones leave, crashed ones restart or break."""
    slots = state["slots"]
    for wid, s in list(slots.items()):
        p = s["proc"]
        if p is None or p.is_alive():
            continue
        s["proc"] = None
        if s["status"] == "retired":
            del slots[wid]
            continue
        reason = exit_reason(p.exitcode)
        if s["status"] == "loading":
            # if the load raised, the collector overwrites this with the real error
            s["status"], s["error"] = "broken", f"morreu ao carregar o modelo: {reason}"
            print(f"[tts] worker {wid} ({s['engine']}) QUEBRADO: {s['error']}", flush=True)
        elif s["status"] == "ready":
            s["restarts"] += 1
            s["last_death"] = reason
            requeue(s, reason)
            if s["restarts"] > MAX_RESTARTS:
                s["status"], s["error"] = "broken", f"morreu {s['restarts']} vezes; último: {reason}"
            else:
                s["status"] = "waiting"
            print(f"[tts] worker {wid} ({s['engine']}) morreu ({reason}); reinício {s['restarts']}/{MAX_RESTARTS}",
                  flush=True)
    for wid, s in list(slots.items()):  # an on-demand engine that broke gives its RAM back
        if s["status"] == "broken" and s["proc"] is None and s["engine"] not in (DEFAULT, ASR):
            engine_failed(s["engine"], s["error"])


def unload_idle():
    now = time.time()
    for e, n in list(state["want"].items()):
        if e == DEFAULT or not n:
            continue
        busy = any(j["engine"] == e and pending_parts(j) for j in jobs.values())
        if not busy and now - state["last_used"].get(e, 0) > IDLE_UNLOAD:
            state["want"][e] = 0
            fit_want()
            print(f"[tts] {e} ocioso há {IDLE_UNLOAD}s: descarregado; {DEFAULT} volta a {state['want'][DEFAULT]}", flush=True)


def reconcile():
    """Adds or retires workers until each engine has what state["want"] asks for."""
    slots, want, pills = state["slots"], state["want"], state["pills"]
    for e in set(want) | {s["engine"] for s in slots.values()} - {ASR}:
        # a broken slot keeps its place (else a worker that dies loading for lack of RAM is recreated forever)
        active = [w for w, s in slots.items() if s["engine"] == e and s["status"] in ("waiting", "loading", "ready", "broken")]
        have, target = len(active) - pills.get(e, 0), want.get(e, 0)
        for _ in range(target - have):
            add_slot(e)
        for _ in range(have - target):
            idle = next((w for st in ("broken", "waiting") for w in active if slots[w]["status"] == st), None)
            if idle is not None:  # holds no process: just drop it
                del slots[idle]
                active.remove(idle)
            else:  # a worker of this engine exits when it takes this from the queue (after the parts ahead of it)
                state["tasks"][e].put(None)
                pills[e] = pills.get(e, 0) + 1


def start_next():
    """One worker loads at a time; on-demand engines first, and only when the RAM budget allows it."""
    slots = state["slots"]
    if any(s["status"] == "loading" for s in slots.values()):
        return
    live = sum(cost(s["engine"]) for s in slots.values()
               if s["engine"] != ASR and s["proc"] is not None and s["proc"].is_alive())
    waiting = sorted((w for w, s in slots.items() if s["status"] == "waiting"),
                     key=lambda w: (slots[w]["engine"] != ASR, slots[w]["engine"] == DEFAULT, w))
    for wid in waiting:
        s = slots[wid]
        if s["engine"] == ASR or live == 0 or live + cost(s["engine"]) <= state["budget"]:
            start_worker(wid, s)
            return


def supervise():
    while not state["stopping"]:
        try:
            with lock:
                reap()
                unload_idle()
                reconcile()
                start_next()
        except Exception:  # a bug here must not silently stop supervision
            traceback.print_exc()
            sys.stderr.flush()
        time.sleep(1)


def collect():
    """Applies worker events to the slots and the in-memory job table."""
    while True:
        ev = state["events"].get()
        kind, wid = ev[0], ev[1]
        with lock:
            s = state["slots"].get(wid)
            if s is None:  # slot already gone (retired); job events below still apply
                s = {"engine": None, "task": None}
            if kind == "ready":
                s["status"], s["device"] = "ready", ev[2] + (f" gpu{s['gpu']}" if s.get("gpu") is not None else "")
                print(f"[tts] worker {wid} ({s['engine']}) pronto ({s['device']})", flush=True)
                continue
            if kind == "broken":
                s["status"], s["error"] = "broken", ev[2]
                print(f"[tts] worker {wid} ({s['engine']}) QUEBRADO: {ev[2]}", flush=True)
                continue
            if kind == "retired":
                s["status"] = "retired"
                state["pills"][s["engine"]] = max(0, state["pills"].get(s["engine"], 0) - 1)
                print(f"[tts] worker {wid} ({s['engine']}) liberado", flush=True)
                continue
            job_id, n = ev[2], ev[3]
            s["task"] = (job_id, n) if kind in ("start", "asr_start") else None
            job = jobs.get(job_id)
            if job is None:  # deleted while a worker was on it
                continue
            if kind == "start":
                job["running"].add(n)
            elif kind == "done":
                job["running"].discard(n)
                job["done"].add(n)
                job["gen_secs"] += ev[4]
                job["audio_secs"] += ev[5]
                state["last_used"][job["engine"]] = time.time()
                if job["transcribe"]:
                    state["tasks"][ASR].put(asr_task(job_id, n))
            elif kind == "fail":
                job["running"].discard(n)
                job["failed"][n] = ev[4]
                print(f"[tts] job {job_id} parte {n} falhou: {ev[4]}", flush=True)
            elif kind == "heard":
                job["heard"][n] = ev[4]
            elif kind == "asr_fail":
                job["failed"][n] = f"transcrição falhou: {ev[4]}"
                print(f"[tts] job {job_id} parte {n}: transcrição falhou: {ev[4]}", flush=True)


@asynccontextmanager
async def lifespan(_app):
    if not TOKEN:
        raise SystemExit("F5_TOKEN não definido")
    for name in INSTALLED:
        engines.get(name)  # unknown name fails at startup, not at the first request
    if DEFAULT not in INSTALLED:
        raise SystemExit(f"F5_DEFAULT_ENGINE={DEFAULT} não está em F5_INSTALLED={INSTALLED}")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    budget = meminfo_mb("MemAvailable") - (ASR_RAM_MB if ASR_MODEL != "off" else 0) - RESERVE_MB
    n_default = auto_workers(budget) if WORKERS == "auto" else int(WORKERS)
    ctx = mp.get_context("spawn")  # CUDA does not survive fork
    state.update(ctx=ctx, events=ctx.Queue(), tasks={name: ctx.Queue() for name in INSTALLED},
                 budget=budget, n_default=n_default, want={DEFAULT: n_default})
    if ASR_MODEL != "off":
        state["tasks"][ASR] = ctx.Queue()
        add_slot(ASR)
    print(f"[tts] instalados {INSTALLED}, padrão {DEFAULT} com {n_default} worker(s) (F5_WORKERS={WORKERS}), "
          f"orçamento de RAM {budget} MB, ASR {ASR_MODEL}; carregando um worker por vez", flush=True)
    threading.Thread(target=collect, daemon=True).start()
    threading.Thread(target=supervise, daemon=True).start()
    yield
    state["stopping"] = True
    with lock:
        for s in state["slots"].values():
            if s["proc"] is not None:
                s["proc"].terminate()


app = FastAPI(title="TTS server", lifespan=lifespan)


def auth(x_token: str = Header("")):
    if not secrets.compare_digest(x_token.encode(), TOKEN.encode()):
        raise HTTPException(401, "token inválido")


def engine_ready(engine):
    return any(s["engine"] == engine and s["status"] == "ready" for s in state["slots"].values())


def job_view(job_id, job):
    total, done = job["total"], len(job["done"])
    if job["failed"]:
        status = "error"
    elif done == total:
        status = "done"
    elif job["running"] or done:
        status = "running"
    else:
        status = "queued"
    out = {
        "job_id": job_id, "engine": job["engine"], "engine_ready": engine_ready(job["engine"]), "status": status,
        "total": total, "done": sorted(job["done"]), "failed": job["failed"],
        "gen_secs": round(job["gen_secs"], 2), "audio_secs": round(job["audio_secs"], 2),
    }
    if job["transcribe"]:
        out["heard"] = dict(sorted(job["heard"].items()))
    return out


def summarize(slots):
    return {
        "workers": len(slots),
        "ready": sum(s["status"] == "ready" for s in slots),
        "loading": sum(s["status"] in ("waiting", "loading") for s in slots),
        "alive": sum(s["proc"] is not None and s["proc"].is_alive() for s in slots),
    }


@app.get("/health", dependencies=[Depends(auth)])
def health():
    with lock:
        all_slots = state["slots"]
        slots = [s for s in all_slots.values() if s["engine"] != ASR]  # TTS workers; the ASR slot has its own section
        pending = sum(pending_parts(j) for j in jobs.values())
        pids = [s["proc"].pid for s in slots if s["proc"] is not None and s["proc"].is_alive()]
        a = asr_slot()
        asr = {"model": ASR_MODEL, "status": a["status"] if a else "off"}
        if a:
            asr.update(device=a["device"], rss_mb=rss_mb(a["proc"].pid) if a["proc"] is not None else None)
        out = {
            "ok": True, "default_engine": DEFAULT, "installed": INSTALLED,
            **summarize(slots),
            "engines": {name: {**summarize([s for s in slots if s["engine"] == name]),
                               "wanted": state["want"].get(name, 0)} for name in INSTALLED},
            "engine_errors": state["engine_errors"],
            "broken": {wid: f"{s['engine']}: {s['error']}" for wid, s in all_slots.items() if s["status"] == "broken"},
            "restarts": {wid: {"engine": s["engine"], "count": s["restarts"], "last": s["last_death"]}
                         for wid, s in all_slots.items() if s["restarts"]},
            "devices": sorted({s["device"] for s in slots if s["status"] == "ready"}),
            "pending_parts": pending,
            "asr": asr,
            "ram_budget_mb": state["budget"], "idle_unload_secs": IDLE_UNLOAD,
        }
    out["ram"] = {"available_mb": meminfo_mb("MemAvailable"), "total_mb": meminfo_mb("MemTotal"),
                  "workers_rss_mb": [rss_mb(pid) for pid in pids]}
    out["gpu"] = gpu_usage()
    return out


@app.post("/jobs", dependencies=[Depends(auth)])
def create_job(body: JobIn):
    engine = body.engine or DEFAULT
    if engine not in INSTALLED:
        raise HTTPException(422, f"motor '{engine}' não está instalado neste server (instalados: {INSTALLED})")
    texts = [t.strip() for t in body.texts]
    if not all(texts):
        raise HTTPException(422, "texto vazio na lista")
    try:
        ref_bytes = base64.b64decode(body.ref_audio, validate=True)
    except binascii.Error:
        raise HTTPException(422, "ref_audio não é base64 válido")
    if not ref_bytes or len(ref_bytes) > MAX_REF_BYTES:
        raise HTTPException(422, "ref_audio vazio ou maior que 10 MB")
    a = asr_slot()
    if body.transcribe and (a is None or a["status"] == "broken"):  # a restarting ASR still drains its queue later
        raise HTTPException(422, f"ASR indisponível neste server (F5_ASR_MODEL={ASR_MODEL}"
                                 f"{', quebrado: ' + str(a['error']) if a else ''})")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    ref = job_dir / "ref.wav"
    ref.write_bytes(ref_bytes)
    opts = {**body.opts, "nfe": body.nfe, "speed": body.speed}
    tasks = [(job_id, n, t, str(ref), body.ref_text.strip(), opts, str(JOBS_DIR)) for n, t in enumerate(texts)]
    with lock:
        jobs[job_id] = {
            "engine": engine, "total": len(texts), "done": set(), "running": set(), "failed": {},
            "retries": {}, "tasks": tasks, "gen_secs": 0.0, "audio_secs": 0.0,
            "transcribe": body.transcribe, "heard": {}, "asr_retries": {},
        }
        request_engine(engine)
    for t in tasks:
        state["tasks"][engine].put(t)
    return {"job_id": job_id, "engine": engine, "engine_ready": engine_ready(engine), "total": len(texts)}


@app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "job não encontrado")
        return job_view(job_id, jobs[job_id])


@app.get("/jobs/{job_id}/parts/{n}", dependencies=[Depends(auth)])
def get_part(job_id: str, n: int):
    with lock:
        job = jobs.get(job_id)
        if job is None or n not in job["done"]:
            raise HTTPException(404, "parte não pronta")
    return FileResponse(JOBS_DIR / job_id / f"{n}.wav", media_type="audio/wav")


@app.delete("/jobs/{job_id}", dependencies=[Depends(auth)])
def delete_job(job_id: str):
    # ponytail: jobs live only until the client deletes them; add a TTL sweep if abandoned jobs fill the disk
    with lock:
        jobs.pop(job_id, None)
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


@app.post("/shutdown", dependencies=[Depends(auth)])
def shutdown(force: bool = False):
    """Stops the server; the notebook's cell 4 sees it exit and releases the machine (Kaggle run ends,
    Colab runtime is unassigned). Refuses while parts are queued unless force."""
    with lock:
        pending = sum(pending_parts(j) for j in jobs.values())
    if pending and not force:
        raise HTTPException(409, f"{pending} partes na fila; espere terminar ou use force")
    # SIGTERM after the response goes out: uvicorn exits cleanly and the lifespan stops the workers
    threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM)).start()
    print(f"[tts] desligando a pedido do client ({pending} partes na fila descartadas)", flush=True)
    return {"shutting_down": True, "discarded_parts": pending}


def main():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
