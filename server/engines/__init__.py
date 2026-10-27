"""TTS engines the server can run. One module per engine, all with the same contract:

    NAME: str                 # id used in the request ("engine") and in F5_ENGINES
    SAMPLE_RATE: int          # sample rate of the arrays synth() returns
    PIP: str                  # extra `pip install` args the Colab notebook runs for this engine
    VENV: bool (optional)     # True = install PIP in its own venv on Colab (pins that clash with other
                              # engines); workers then run under that venv's Python. Default False.
    RAM_MB: int (optional)    # host RAM one loaded worker takes; the supervisor packs engines into the
                              # RAM budget with it when loading on demand. Default 2500.

    load(device) -> model     # once per worker process; device is "cuda", "cpu" or None (= auto)
    warmup(model) -> None     # tiny end-to-end generation proving the pipeline works; raise on failure
    synth(model, ref_wav, ref_text, text, opts) -> numpy float32 mono array at SAMPLE_RATE

`ref_wav` is a path to the reference clip (5-12 s, any sample rate), `ref_text` its exact
transcript, `text` one narration part in pt-BR. `opts` is the request's free-form dict:
"nfe" and "speed" are always present (an engine may ignore them); anything else is
engine-specific (e.g. chatterbox "exaggeration"). Text normalization an engine needs
(e.g. lowercase) belongs in its synth().

Engines are imported lazily inside the worker process, so a module must not import its
heavy dependencies at module level.
"""

import importlib

AVAILABLE = ("f5", "qwen3", "chatterbox")


def get(name):
    if name not in AVAILABLE:
        raise ValueError(f"motor desconhecido: {name} (disponíveis: {', '.join(AVAILABLE)})")
    return importlib.import_module(f"engines.{name}")
