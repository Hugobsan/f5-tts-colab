"""F5-TTS with the pt-BR checkpoint from firstpixel/F5-TTS-pt-br (Portuguese only, CC-BY-NC)."""

import os

NAME = "f5"
SAMPLE_RATE = 24000
PIP = "f5-tts==1.1.22"
MODEL = os.environ.get("F5_MODEL", "F5TTS_Base")
CKPT = os.environ.get("F5_CKPT", "/content/model/model_last.safetensors")


def load(device):
    from f5_tts.api import F5TTS

    return F5TTS(model=MODEL, ckpt_file=CKPT, device=device)


def warmup(tts):
    from importlib.resources import files

    ref = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    tts.infer(ref, "some call me nature, others call me mother nature.", "ok.",
              nfe_step=4, show_info=lambda *a: None, progress=None)


def synth(tts, ref_wav, ref_text, text, opts):
    # the pt-BR checkpoint was trained on lowercase text
    wav, _, _ = tts.infer(ref_wav, ref_text.strip().lower(), text.lower(),
                          nfe_step=int(opts.get("nfe", 32)), speed=float(opts.get("speed", 1.0)),
                          show_info=lambda *a: None, progress=None)
    return wav
