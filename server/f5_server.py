"""TTS server: HTTP job queue served by GPU worker processes, one or more TTS engines.

Each job is a list of texts (one per narration paragraph) for one engine. The texts go
into that engine's queue and its workers pick them up in parallel, so a job finishes in
roughly (total / workers) time. Every request answers in milliseconds: the client
submits, polls and downloads, so nothing hits Cloudflare's 100 s limit.

Workers load one at a time (loading peaks RAM) and a supervisor replaces any worker
that dies after becoming ready, putting its in-flight part back in the queue. A worker
that dies while loading is not retried: that is almost always lack of RAM, and it shows
up in /health with the exit code.

One more slot, "asr", runs faster-whisper (worker.asr_main) under the same supervisor: a job
sent with transcribe=true gets each finished part queued for it, and the text comes back in
the job status as heard{n: text}, so the client checks parts without its own Whisper.

Run:  F5_TOKEN=<secret> python run.py
Config (env):
  F5_TOKEN            required
  F5_ENGINES          engines and worker counts, first = default: "f5:auto", "f5:3,qwen3:1,chatterbox:1"
                      (count "auto" = as many as free RAM/VRAM allow, sized for F5; no count = 1)
  F5_WORKERS          legacy: same as F5_ENGINES="f5:<value>" when F5_ENGINES is unset
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
ENGINES = os.environ.get("F5_ENGINES") or f"f5:{os.environ.get('F5_WORKERS', 'auto')}"
ASR_MODEL = os.environ.get("F5_ASR_MODEL", "small")
ASR = "asr"  # slot "engine" name of the ASR worker
JOBS_DIR = Path(os.environ.get("F5_JOBS_DIR", "/content/jobs"))
PORT = int(os.environ.get("PORT", "8000"))
MAX_REF_BYTES = 10 * 1024 * 1024
# ponytail: per-worker cost from one Colab T4 run of F5 (~0.9 GB VRAM idle; 6 workers on 12.7 GB RAM lost 2).
# Recalibrate from /health ram.workers_rss_mb and gpu.mem_used_mb if auto picks badly.
RAM_PER_WORKER_MB = 2500
VRAM_PER_WORKER_MB = 2000
# ponytail: estimate for faster-whisper small (float16) + CUDA libs, not measured on Colab; check /health asr.rss_mb
ASR_RAM_MB = 1500
MAX_AUTO_WORKERS = 8
MAX_RESTARTS = 5


class JobIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=2000)
    ref_audio: str = Field(description="reference wav, base64")
    ref_text: str = Field(min_length=1)
    engine: str | None = Field(None, description="engine id; default = first in F5_ENGINES")
    nfe: int = Field(32, ge=1, le=64)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    opts: dict = Field(default_factory=dict, description="engine-specific options")
    transcribe: bool = Field(False, description="transcribe each finished part with the ASR worker (status heard{})")


# slot status: waiting -> loading -> ready (-> waiting again if it dies) | broken
state = {"slots": [], "engines": {}, "default": None, "ctx": None, "tasks": {}, "events": None, "stopping": False}
jobs: dict[str, dict] = {}
lock = threading.Lock()


def new_slot(engine):
    return {"engine": engine, "proc": None, "status": "waiting", "device": None, "task": None,
            "error": None, "restarts": 0, "last_death": None}


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


def gpu_usage():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return None
    util, used, total = (int(x) for x in out.split(", "))
    return {"util_pct": util, "mem_used_mb": used, "mem_total_mb": total}


def auto_workers():
    """As many workers as free RAM and VRAM allow; 1 without a GPU."""
    gpu = gpu_usage()
    if gpu is None:
        return 1
    by_ram = (meminfo_mb("MemAvailable") - (ASR_RAM_MB if ASR_MODEL != "off" else 0)) // RAM_PER_WORKER_MB
    by_vram = (gpu["mem_total_mb"] - gpu["mem_used_mb"]) // VRAM_PER_WORKER_MB
    return max(1, min(by_ram, by_vram, MAX_AUTO_WORKERS))


def parse_engines(spec):
    """'f5:auto,qwen3:1' -> {'f5': n, 'qwen3': 1} in order. 'auto' is only sized for a single engine."""
    out = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, count = item.partition(":")
        engines.get(name)  # unknown name fails at startup, not at the first request
        out[name] = count or "1"
    if not out:
        raise SystemExit("F5_ENGINES vazio")
    for name, count in out.items():
        out[name] = (auto_workers() if len(out) == 1 else 1) if count == "auto" else int(count)
    return out


def exit_reason(code):
    if code == -signal.SIGKILL:
        return f"exit {code} (SIGKILL: quase sempre falta de RAM)"
    return f"exit {code}"


def asr_task(job_id, n):
    return job_id, n, str(JOBS_DIR / job_id / f"{n}.wav")


def asr_slot():
    return next((s for s in state["slots"] if s["engine"] == ASR), None)


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
    s["proc"].start()
    s["status"] = "loading"


def supervise():
    """Starts workers one at a time and replaces the ones that die after ready."""
    while not state["stopping"]:
        with lock:
            for wid, s in enumerate(state["slots"]):
                p = s["proc"]
                if p is None or p.is_alive():
                    continue
                s["proc"] = None
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
            if not any(s["status"] == "loading" for s in state["slots"]):
                for wid, s in enumerate(state["slots"]):
                    if s["status"] == "waiting":
                        start_worker(wid, s)
                        break
        time.sleep(1)


def collect():
    """Applies worker events to the slots and the in-memory job table."""
    while True:
        ev = state["events"].get()
        kind, wid = ev[0], ev[1]
        with lock:
            s = state["slots"][wid]
            if kind == "ready":
                s["status"], s["device"] = "ready", ev[2]
                print(f"[tts] worker {wid} ({s['engine']}) pronto ({ev[2]})", flush=True)
                continue
            if kind == "broken":
                s["status"], s["error"] = "broken", ev[2]
                print(f"[tts] worker {wid} ({s['engine']}) QUEBRADO: {ev[2]}", flush=True)
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
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    counts = parse_engines(ENGINES)
    ctx = mp.get_context("spawn")  # CUDA does not survive fork
    tasks = {name: ctx.Queue() for name in counts}
    slots = [new_slot(name) for name, n in counts.items() for _ in range(n)]
    if ASR_MODEL != "off":
        tasks[ASR] = ctx.Queue()
        slots.insert(1, new_slot(ASR))  # right after the first TTS worker, so transcripts are available early
    state.update(ctx=ctx, engines=counts, default=next(iter(counts)), events=ctx.Queue(), tasks=tasks, slots=slots)
    print(f"[tts] motores {counts} (F5_ENGINES={ENGINES}), ASR {ASR_MODEL}, carregando um worker por vez", flush=True)
    threading.Thread(target=collect, daemon=True).start()
    threading.Thread(target=supervise, daemon=True).start()
    yield
    state["stopping"] = True
    with lock:
        for s in state["slots"]:
            if s["proc"] is not None:
                s["proc"].terminate()


app = FastAPI(title="TTS server", lifespan=lifespan)


def auth(x_token: str = Header("")):
    if not secrets.compare_digest(x_token.encode(), TOKEN.encode()):
        raise HTTPException(401, "token inválido")


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
        "job_id": job_id, "engine": job["engine"], "status": status, "total": total, "done": sorted(job["done"]),
        "failed": job["failed"], "gen_secs": round(job["gen_secs"], 2), "audio_secs": round(job["audio_secs"], 2),
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
        slots = [s for s in state["slots"] if s["engine"] != ASR]  # TTS workers; the ASR slot has its own section
        pending = sum(j["total"] - len(j["done"] | j["failed"].keys()) for j in jobs.values())
        pids = [s["proc"].pid for s in slots if s["proc"] is not None and s["proc"].is_alive()]
        a = asr_slot()
        asr = {"model": ASR_MODEL, "status": a["status"] if a else "off"}
        if a:
            asr.update(device=a["device"], rss_mb=rss_mb(a["proc"].pid) if a["proc"] is not None else None)
        out = {
            "ok": True, "default_engine": state["default"], "engines_spec": ENGINES,
            **summarize(slots),
            "engines": {name: summarize([s for s in slots if s["engine"] == name]) for name in state["engines"]},
            "broken": {wid: f"{s['engine']}: {s['error']}" for wid, s in enumerate(state["slots"]) if s["status"] == "broken"},
            "restarts": {wid: {"engine": s["engine"], "count": s["restarts"], "last": s["last_death"]}
                         for wid, s in enumerate(state["slots"]) if s["restarts"]},
            "devices": sorted({s["device"] for s in slots if s["status"] == "ready"}),
            "pending_parts": pending,
            "asr": asr,
        }
    out["ram"] = {"available_mb": meminfo_mb("MemAvailable"), "total_mb": meminfo_mb("MemTotal"),
                  "workers_rss_mb": [rss_mb(pid) for pid in pids]}
    out["gpu"] = gpu_usage()
    return out


@app.post("/jobs", dependencies=[Depends(auth)])
def create_job(body: JobIn):
    engine = body.engine or state["default"]
    if engine not in state["engines"]:
        raise HTTPException(422, f"motor '{engine}' não está carregado neste server (carregados: {list(state['engines'])})")
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
    for t in tasks:
        state["tasks"][engine].put(t)
    return {"job_id": job_id, "engine": engine, "total": len(texts)}


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


def main():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
