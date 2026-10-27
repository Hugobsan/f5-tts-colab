"""Qwen3-TTS Base (voice cloning) from QwenLM/Qwen3-TTS, Apache-2.0, multilingual incl. Portuguese.

Model: QWEN3_MODEL env (default Qwen/Qwen3-TTS-12Hz-1.7B-Base; Qwen/Qwen3-TTS-12Hz-0.6B-Base is
the smaller one). QWEN3_DTYPE env forces the weights dtype (float32 / float16 / bfloat16);
default bfloat16 on GPUs that support it, float16 on older ones (T4), float32 on CPU.

opts (all optional besides the always-present ones):
    nfe            ignored (autoregressive model, no diffusion steps)
    speed          != 1.0 time-stretches the output with librosa (the model has no speed control)
    language       "portuguese" (default) or any name from model.get_supported_languages()
    temperature    sampling temperature, default 0.9 (lower = steadier, flatter)
    top_k          default 50
    top_p          default 1.0
    repetition_penalty  default 1.05
    seed           int; makes the sampling reproducible
    xvec_only      true = clone from the speaker embedding only (ignores ref_text, weaker clone)
    max_new_tokens codec tokens cap, default 2048 (12 tokens = 1 s of audio)
"""

import functools
import os

NAME = "qwen3"
SAMPLE_RATE = 24000
RAM_MB = 6000  # estimate: 11.8 GB peak on CPU fp32; fp16 on GPU should need about half, not measured on Colab
PIP = "qwen-tts==0.1.1"
MODEL = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


def load(device):
    import torch
    from qwen_tts import Qwen3TTSModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = os.environ.get("QWEN3_DTYPE")
    if dtype:
        dtype = getattr(torch, dtype)
    elif device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    else:
        dtype = torch.float32
    return Qwen3TTSModel.from_pretrained(MODEL, device_map=device, dtype=dtype)


def warmup(tts):
    import numpy as np

    noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.05).astype(np.float32)
    wavs, _ = tts.generate_voice_clone(text="ok.", language="english", ref_audio=(noise, SAMPLE_RATE),
                                       x_vector_only_mode=True, max_new_tokens=24)
    if not len(wavs[0]):
        raise RuntimeError("qwen3 warmup produced no audio")


@functools.lru_cache(maxsize=8)
def _prompt(tts, ref_wav, ref_text, xvec_only):
    # the reference is the same for every part of a narration: encode it once
    return tts.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text, x_vector_only_mode=xvec_only)


def synth(tts, ref_wav, ref_text, text, opts):
    import numpy as np

    if opts.get("seed") is not None:
        import torch
        torch.manual_seed(int(opts["seed"]))
    gen = {k: opts[k] for k in ("temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens")
           if opts.get(k) is not None}
    prompt = _prompt(tts, ref_wav, ref_text.strip(), bool(opts.get("xvec_only", False)))
    wavs, _ = tts.generate_voice_clone(text=text.strip(), language=opts.get("language", "portuguese"),
                                       voice_clone_prompt=prompt, **gen)
    wav = np.asarray(wavs[0], dtype=np.float32)
    speed = float(opts.get("speed", 1.0))
    if speed != 1.0:
        import librosa
        wav = librosa.effects.time_stretch(wav, rate=speed)
    return wav
