"""Worker process: loads one engine, warms it up, then generates parts from that engine's queue.

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
        job_id, n, text, ref, ref_text, opts, jobs_dir = tasks.get()
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
