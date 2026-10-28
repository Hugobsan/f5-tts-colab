"""Python client for the F5-TTS server (stdlib only)."""

import base64
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "f5tts" / "config.json"
WHISPER_MODEL = Path(os.environ.get("F5_WHISPER_MODEL", Path.home() / ".cache/hyperframes/whisper/models/ggml-small.bin"))


class F5Error(RuntimeError):
    pass


class ServerDown(F5Error):
    pass


def load_config():
    """URL and token: F5_URL/F5_TOKEN env vars override ~/.config/f5tts/config.json."""
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    url = os.environ.get("F5_URL") or cfg.get("url")
    token = os.environ.get("F5_TOKEN") or cfg.get("token")
    return url, token


def save_config(**changes):
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    cfg.update({k: v for k, v in changes.items() if v is not None})
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    CONFIG_PATH.chmod(0o600)
    return cfg


def load_pronunciation(path):
    """TSV `termo<TAB>como falar`, '#' comments. Longest terms apply first ("pull requests" before "pull request")."""
    rules = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        term, spoken = line.split("\t", 1)
        rules.append((term.strip(), spoken.strip()))
    rules.sort(key=lambda r: -len(r[0]))
    return [(re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE), s) for t, s in rules]


def apply_pronunciation(text, rules):
    for rx, spoken in rules or ():
        text = rx.sub(spoken, text)
    return text


def words(text):
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", t)


# Whisper writes "2026" where the script says "dois mil e vinte e seis": numbers (spelled or digits) and
# the connective "e" are left out of the comparison on both sides.
NUMBER_WORDS = set("""e zero um uma dois duas tres quatro cinco seis sete oito nove dez onze doze treze quatorze
catorze quinze dezesseis dezessete dezoito dezenove vinte trinta quarenta cinquenta sessenta setenta oitenta noventa
cem cento duzentos duzentas trezentos trezentas quatrocentos quatrocentas quinhentos quinhentas seiscentos seiscentas
setecentos setecentas oitocentos oitocentas novecentos novecentas mil milhao milhoes bilhao bilhoes""".split())


def content_words(text):
    return [w for w in words(text) if w not in NUMBER_WORDS and not w.isdigit()]


def match_score(expected, heard):
    """Share of the expected words found, in order, in what was heard (1.0 = all)."""
    e, h = content_words(expected), content_words(heard)
    blocks = difflib.SequenceMatcher(None, e, h, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / max(1, len(e))


def whisper_transcribe(wav_bytes, model=WHISPER_MODEL):
    """Local whisper.cpp transcription (pt) of a wav blob."""
    with tempfile.TemporaryDirectory() as tmp:
        src, wav16 = Path(tmp) / "in.wav", Path(tmp) / "16k.wav"
        src.write_bytes(wav_bytes)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "16000", "-ac", "1", str(wav16)], check=True)
        out = subprocess.run(["whisper-cli", "-m", str(model), "-l", "pt", "-nt", "-np", "-t", "2", "-f", str(wav16)],
                             capture_output=True, text=True, check=True).stdout
    return " ".join(out.split())


def server_asr(health):
    """The /health `asr` section when the server can transcribe parts (faster-whisper on its GPU), else None.

    None on servers without ASR (older ones, or F5_ASR_MODEL=off) and while it is loading or broken.
    """
    asr = health.get("asr") or {}
    return asr if asr.get("status") == "ready" else None


class F5Client:
    def __init__(self, url=None, token=None, timeout=30, retries=3):
        cfg_url, cfg_token = load_config()
        self.url = (url or cfg_url or "").rstrip("/")
        self.token = token or cfg_token
        if not self.url or not self.token:
            raise F5Error("URL/token não configurados: rode `f5tts config --url ... --token ...`")
        self.timeout = timeout
        self.retries = retries

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path, data=data, method=method,
            headers={"x-token": self.token, "content-type": "application/json", "user-agent": "f5tts-client"},
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                # 5xx here is usually the tunnel hiccuping (502/524/530): retry
                if e.code < 500 or attempt == self.retries:
                    raise F5Error(f"{method} {path} -> HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise ServerDown(f"server inacessível em {self.url}: {e}") from None
            time.sleep(2 ** attempt)

    def _json(self, method, path, body=None):
        return json.loads(self._req(method, path, body))

    def health(self):
        try:
            return self._json("GET", "/health")
        except ServerDown:
            raise
        except F5Error as e:
            if "HTTP 5" in str(e):  # tunnel up but no origin behind it (530/1033) or origin erroring
                raise ServerDown(f"server fora do ar ({self.url}): {e}") from None
            raise

    def ensure_ready(self, engine=None):
        h = self.health()
        installed = h.get("installed", list(h.get("engines", {})))  # older servers list only loaded engines
        if engine and installed:
            if engine not in installed:
                raise F5Error(f"modelo '{engine}' não está instalado no server (instalados: {installed})")
            if "installed" in h and engine != h["default_engine"]:
                return h  # on-demand engine: the server loads it when the job arrives
            h = {**h, **h["engines"][engine]}
        if h["ready"] == 0:
            if h["broken"]:
                raise ServerDown(f"todos os workers falharam ao carregar: {h['broken']}")
            raise ServerDown(f"server no ar mas nenhum worker pronto ainda ({h['loading']}/{h['workers']} carregando, um por vez)")
        return h

    def submit(self, texts, ref_audio, ref_text, nfe=32, speed=1.0, engine=None, opts=None, transcribe=False):
        """engine: server engine id (None = server default); opts: engine-specific extras.
        transcribe: the server's ASR transcribes each finished part (status()["heard"]{"n": text})."""
        body = {
            "texts": list(texts), "ref_text": ref_text, "nfe": nfe, "speed": speed,
            "ref_audio": base64.b64encode(Path(ref_audio).read_bytes()).decode(),
        }
        if engine:
            body["engine"] = engine
        if opts:
            body["opts"] = opts
        if transcribe:
            body["transcribe"] = True
        return self._json("POST", "/jobs", body)

    def status(self, job_id):
        return self._json("GET", f"/jobs/{job_id}")

    def part(self, job_id, n):
        return self._req("GET", f"/jobs/{job_id}/parts/{n}")

    def delete(self, job_id):
        return self._json("DELETE", f"/jobs/{job_id}")

    def shutdown(self, force=False):
        """Stops the server and frees the Colab/Kaggle machine; HTTP 409 (F5Error) while parts are queued."""
        return self._json("POST", "/shutdown" + ("?force=true" if force else ""))

    def synthesize(self, parts, out_wav, refs, nfe=32, speed=1.0, pause=0.25, sample_rate=None,
                   pronunciation=None, validate=None, min_score=0.9, max_attempts=3, engine=None, opts=None,
                   poll=2.0, stall_timeout=900, on_progress=None):
        """Generates every part, joins them in order with `pause` seconds of silence and writes out_wav.

        parts: [(style, text)]; refs: {style: (ref_wav, ref_text)}. Each style is its own job, all in parallel.
        engine/opts: server engine id (None = server default) and its engine-specific options.
        pronunciation: rules from load_pronunciation, applied to the text sent (never to the text checked).
        validate: callable wav_bytes -> heard text (e.g. whisper_transcribe), turns checking on. When the
        server has its own ASR ready (see server_asr) the parts are transcribed there instead and validate
        is only the fallback; the report's "asr" says which ("server" | "local"). Parts whose match_score
        with the original text is below min_score are regenerated (new random seed) up to max_attempts; the
        best attempt is kept. Raises ServerDown/F5Error on failure; returns a per-part report.
        """
        parts = [(style, text.strip()) for style, text in parts if text.strip()]
        if not parts:
            raise F5Error("nenhum texto para gerar")
        unknown = {style for style, _ in parts} - refs.keys()
        if unknown:
            raise F5Error(f"estilo sem ref: {sorted(unknown)} (disponíveis: {sorted(refs)})")
        spoken = [apply_pronunciation(text, pronunciation) for _, text in parts]
        on_server = validate is not None and server_asr(self.ensure_ready(engine)) is not None
        t0 = time.time()
        jobs = {}  # job_id -> {n: part index} still to download
        attempts = [0] * len(parts)
        best = [None] * len(parts)  # (score, wav, heard)
        checking = {}  # future -> (part index, wav)
        retry = []
        pool = ThreadPoolExecutor(2) if validate and not on_server else None

        def send(indexes):
            by_style = {}
            for i in indexes:
                by_style.setdefault(parts[i][0], []).append(i)
            for style, ii in by_style.items():
                job_id = self.submit([spoken[i] for i in ii], *refs[style], nfe, speed, engine, opts,
                                     transcribe=on_server)["job_id"]
                jobs[job_id] = dict(enumerate(ii))
                for i in ii:
                    attempts[i] += 1

        def keep(i, wav, score, heard):
            if best[i] is None or score > best[i][0]:
                best[i] = (score, wav, heard)

        def check(i, wav, heard):
            # dictionary on both sides: "deploy" heard counts like "diplói", "comit" like "comít"
            score = match_score(spoken[i], apply_pronunciation(heard, pronunciation))
            keep(i, wav, score, heard)
            if score < min_score and attempts[i] < max_attempts:
                retry.append(i)

        send(range(len(parts)))
        last_change = time.time()
        loading = False  # engine still loading on the server (on-demand engines)
        try:
            while jobs or checking:
                loading = False
                for job_id, todo in list(jobs.items()):
                    st = self.status(job_id)
                    loading = loading or st.get("engine_ready") is False
                    if loading:
                        last_change = time.time()  # loading an engine is progress, not a stall
                    if st["failed"]:
                        raise F5Error(f"partes falharam no server: {st['failed']}")
                    heard = st.get("heard", {})  # JSON keys: "n"
                    for n in st["done"]:
                        if n in todo and (not on_server or str(n) in heard):
                            i, wav = todo.pop(n), self.part(job_id, n)
                            last_change = time.time()
                            if on_server:
                                check(i, wav, heard[str(n)])
                            elif pool:
                                checking[pool.submit(validate, wav)] = (i, wav)
                            else:
                                keep(i, wav, 1.0, None)
                    if not todo:
                        del jobs[job_id]
                        self._forget(job_id)
                for fut in [f for f in checking if f.done()]:
                    i, wav = checking.pop(fut)
                    check(i, wav, fut.result())
                if retry:
                    send(retry)
                    retry.clear()
                    last_change = time.time()
                if on_progress:
                    busy = {i for todo in jobs.values() for i in todo.values()} | {i for i, _ in checking.values()}
                    on_progress({"done": sum(b is not None and i not in busy for i, b in enumerate(best)),
                                 "total": len(parts), "retries": sum(attempts) - len(parts),
                                 "loading": engine if loading else None})
                if time.time() - last_change > stall_timeout:
                    raise F5Error(f"sem progresso há {stall_timeout}s (worker morto? veja `f5tts status`)")
                if jobs or checking:
                    time.sleep(poll if jobs else 0.2)
        finally:
            for job_id in jobs:
                self._forget(job_id)
            if pool:
                pool.shutdown(wait=False, cancel_futures=True)

        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        join_wavs([b[1] for b in best], out_wav, pause, sample_rate)
        report = [{"n": i, "style": parts[i][0], "text": parts[i][1], "attempts": attempts[i],
                   "score": round(b[0], 3), "heard": b[2]} for i, b in enumerate(best)]
        return {"total": len(parts), "wall_secs": round(time.time() - t0, 2),
                "asr": ("server" if on_server else "local") if validate else None,
                "retries": sum(attempts) - len(parts), "parts": report,
                "below_min": [r for r in report if r["score"] < min_score]}

    def _forget(self, job_id):
        try:
            self.delete(job_id)
        except F5Error:
            pass


def join_wavs(blobs, out_wav, pause=0.25, sample_rate=None):
    """Concatenates wav byte blobs (same format) with silence between them; resamples via ffmpeg if asked."""
    with tempfile.TemporaryDirectory() as tmp:
        joined = Path(tmp) / "joined.wav"
        with wave.open(str(joined), "wb") as out:
            for i, blob in enumerate(blobs):
                part = Path(tmp) / f"{i}.wav"
                part.write_bytes(blob)
                with wave.open(str(part), "rb") as w:
                    if i == 0:
                        out.setparams(w.getparams())
                        silence = b"\x00" * int(w.getframerate() * pause) * w.getsampwidth() * w.getnchannels()
                    else:
                        out.writeframes(silence)
                    out.writeframes(w.readframes(w.getnframes()))
        if sample_rate:
            if not shutil.which("ffmpeg"):
                raise F5Error("ffmpeg não encontrado (necessário para --sr)")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-ar", str(sample_rate), "-ac", "1", str(out_wav)], check=True)
        else:
            shutil.move(joined, out_wav)


def wav_duration(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()
