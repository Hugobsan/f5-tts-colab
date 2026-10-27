"""Resemble AI Chatterbox Multilingual (package chatterbox-tts, MIT; weights MIT), language_id="pt".

Voice cloning is zero-shot from the `ref_wav` audio prompt alone: `ref_text` is NOT used
(Chatterbox conditions on the reference audio's speaker embedding and speech tokens; the
first 6 s feed the T3 prompt, the first 10 s the S3Gen decoder).

Checkpoint (env CHATTERBOX_CHECKPOINT):
  "pt-br" (default)  ResembleAI/Chatterbox-Multilingual-pt-br: Brazilian Portuguese finetune
                     (t3_pt_br.safetensors + s3gen_v3.pt), plus ve.pt and the tokenizer from
                     ResembleAI/chatterbox. Same loading recipe as ResembleAI's pt-br demo Space.
  "base"             ResembleAI/chatterbox 23-language model as the package ships it
                     (ChatterboxMultilingualTTS.from_pretrained: t3_mtl23ls_v2 + s3gen.pt).

opts ("nfe" and "speed" are ignored: autoregressive T3 + fixed-step S3Gen, no speed knob):
  exaggeration        0.5   emotion/intensity; >0.5 more expressive (and usually faster)
  cfg_weight          0.5   classifier-free guidance / pace; lower = slower, more deliberate
  temperature         0.8
  repetition_penalty  2.0
  min_p               0.05
  top_p               1.0
  seed                None  int -> torch.manual_seed before generating (reproducible takes)
Defaults are the package's generate() defaults.

Output is 24 kHz and carries Resemble's inaudible Perth watermark (applied by the package).
"""

import os

NAME = "chatterbox"
SAMPLE_RATE = 24000
# --no-deps: chatterbox-tts pins torch==2.6.0, transformers==5.2.0 and gradio==6.8.0, which would
# downgrade Colab's torch and clash with f5-tts (gradio>=6.15). These are its runtime imports.
# setuptools<81: resemble-perth imports pkg_resources; without it perth.PerthImplicitWatermarker is
# None and ChatterboxMultilingualTTS() fails with "'NoneType' object is not callable".
PIP = ("--no-deps chatterbox-tts==0.1.7 s3tokenizer conformer==0.3.2 resemble-perth "
       "diffusers==0.29.0 omegaconf pyloudnorm spacy-pkuseg pykakasi==2.3.0 setuptools<81 "
       "onnx ml-dtypes srsly catalogue importlib-metadata zipp jaconv deprecated wrapt")  # their deps
CHECKPOINT = os.environ.get("CHATTERBOX_CHECKPOINT", "pt-br")
LANG = "pt"
BASE_REPO = "ResembleAI/chatterbox"
PTBR_REPO = "ResembleAI/Chatterbox-Multilingual-pt-br"

_GEN_KEYS = ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")


def load(device):
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if CHECKPOINT == "base":
        return ChatterboxMultilingualTTS.from_pretrained(device)
    if CHECKPOINT != "pt-br":
        raise ValueError(f"CHATTERBOX_CHECKPOINT inválido: {CHECKPOINT} (use pt-br ou base)")

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(hf_hub_download(BASE_REPO, "ve.pt"), map_location="cpu", weights_only=True))
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_file(hf_hub_download(PTBR_REPO, "t3_pt_br.safetensors")))
    s3gen = S3Gen()
    # non-strict like the demo Space: s3gen_v3 lacks only tokenizer._mel_filters/window (built in __init__)
    s3gen.load_state_dict(torch.load(hf_hub_download(PTBR_REPO, "s3gen_v3.pt"), map_location="cpu",
                                     weights_only=True), strict=False)
    tok = MTLTokenizer(hf_hub_download(PTBR_REPO, "grapheme_mtl_merged_expanded_v1.json"))
    return ChatterboxMultilingualTTS(t3.to(device).eval(), s3gen.to(device).eval(), ve.to(device).eval(),
                                     tok, device)


def warmup(tts):
    import tempfile

    import numpy as np
    import soundfile as sf
    from chatterbox.models.s3gen import S3GEN_SR

    # no bundled ref clip for the pt-br loader: 2 s of a quiet tone exercises the whole pipeline
    t = np.arange(2 * S3GEN_SR) / S3GEN_SR
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), S3GEN_SR)
        wav = tts.generate("Oi.", language_id=LANG, audio_prompt_path=f.name)
    if wav.numel() == 0:
        raise RuntimeError("chatterbox warmup gerou áudio vazio")


def synth(tts, ref_wav, ref_text, text, opts):
    import torch

    if opts.get("seed") is not None:
        torch.manual_seed(int(opts["seed"]))
    kw = {k: float(opts[k]) for k in _GEN_KEYS if opts.get(k) is not None}
    wav = tts.generate(text, language_id=LANG, audio_prompt_path=ref_wav, **kw)
    return wav.squeeze(0).cpu().numpy().astype("float32")
