"""Worker process: loads one engine, warms it up, then generates parts from that engine's queue.
asr_main is the ASR worker: transcribes finished parts with faster-whisper.

Imports only stdlib + numpy at module level, because a worker may run under a different
Python (the engine's own venv on Colab, see ENGINE_PYTHON_<name> in f5_server.py).
"""

import os
import sys
import time
import wave
from pathlib import Path


def isolate_site_packages():
    """spawn hands the child the parent's sys.path; under a venv interpreter keep only the venv's packages."""
    import site

    if sys.prefix == sys.base_prefix:  # same interpreter as the parent: nothing to isolate
        return
    own = site.getsitepackages()
    sys.path[:] = [p for p in sys.path if "site-packages" not in p and "dist-packages" not in p] + own


def write_wav(path, samples, sample_rate):
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return len(pcm) / sample_rate


def worker_main(wid, engine_name, tasks, events):
    isolate_site_packages()
    try:
        import engines

        eng = engines.get(engine_name)
        model = eng.load(os.environ.get("F5_DEVICE") or None)
        eng.warmup(model)
    except Exception as e:
        events.put(("broken", wid, f"{type(e).__name__}: {e}"))
        return
    events.put(("ready", wid, str(getattr(model, "device", None) or os.environ.get("F5_DEVICE") or "auto")))

    while True:
        task = tasks.get()
        if task is None:  # the supervisor is shrinking this engine's pool: free the RAM
            events.put(("retired", wid))
            return
        job_id, n, text, ref, ref_text, opts, jobs_dir = task
        out = Path(jobs_dir) / job_id / f"{n}.wav"
        if not out.parent.exists():  # job deleted (cancelled) while queued
            continue
        events.put(("start", wid, job_id, n))
        t0 = time.time()
        try:
            secs = write_wav(out, eng.synth(model, ref, ref_text, text, opts), eng.SAMPLE_RATE)
            events.put(("done", wid, job_id, n, time.time() - t0, secs))
        except Exception as e:
            events.put(("fail", wid, job_id, n, f"{type(e).__name__}: {e}"))


def preload_cuda_libs():
    """ctranslate2 dlopens cuBLAS/cuDNN by soname, but pip's nvidia-* wheels (where Colab's torch gets
    them) are not on the loader path. Loading them by full path first makes those dlopens succeed."""
    import ctypes
    import glob
    import site

    libs = [lib for sp in site.getsitepackages() for pattern in ("nvidia/cublas/lib/lib*.so*", "nvidia/cudnn/lib/lib*.so*")
            for lib in sorted(glob.glob(f"{sp}/{pattern}"))]
    for _ in range(3):  # cuDNN sub-libraries depend on each other: retry the ones that failed
        failed = []
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if not failed:
            break
        libs = failed


def load_asr(model_name):
    """faster-whisper on cuda/float16, falling back to cpu/int8. Warmed up, because a missing CUDA
    library (cuDNN, cuBLAS) only fails at the first transcription, not at load.
    Returns (model, device); device says why cuda was not used when it fell back."""
    import numpy as np
    from faster_whisper import WhisperModel

    options = [("cpu", "int8")] if os.environ.get("F5_DEVICE") == "cpu" else [("cuda", "float16"), ("cpu", "int8")]
    if options[0][0] == "cuda":
        preload_cuda_libs()
    why = ""
    for device, compute in options:
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
            list(model.transcribe(np.zeros(16000, np.float32), language="pt")[0])
            return model, device + why
        except Exception as e:
            if device == options[-1][0]:
                raise
            why = f" (cuda falhou: {type(e).__name__}: {str(e)[:160]})"
            print(f"[asr] {device} indisponível ({type(e).__name__}: {e}); tentando cpu", flush=True)


def asr_main(wid, model_name, tasks, events):
    """ASR worker: transcribes finished parts (pt) so the client can check them without a local Whisper.

    Runs in the server's own env (faster-whisper comes from server/requirements.txt), never an engine venv.
    """
    try:
        model, device = load_asr(model_name)
    except Exception as e:
        events.put(("broken", wid, f"{type(e).__name__}: {e}"))
        return
    events.put(("ready", wid, device))

    while True:
        job_id, n, wav = tasks.get()
        if not os.path.exists(wav):  # job deleted while queued
            continue
        events.put(("asr_start", wid, job_id, n))
        try:
            segments, _ = model.transcribe(wav, language="pt")
            events.put(("heard", wid, job_id, n, " ".join(" ".join(s.text for s in segments).split())))
        except Exception as e:
            events.put(("asr_fail", wid, job_id, n, f"{type(e).__name__}: {e}"))
