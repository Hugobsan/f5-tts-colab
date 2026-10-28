# Notebook cell 2: `%run -i /content/repo/colab/setup.py`
# Installs the server deps and every engine in INSTALL (defined in cell 1). Engines are only
# installed here; the server loads them on demand. Defines ENGINE_PYTHON for start.py.
import ast
import subprocess
import sys
from pathlib import Path


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def engine_meta(path):
    """NAME/PIP/VENV constants of an engine module, read without importing it."""
    consts = {t.id: ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and t.id in ("NAME", "PIP", "VENV")}
    return consts["NAME"], {"pip": consts["PIP"], "venv": consts.get("VENV", False)}


ENGINE_META = dict(engine_meta(p) for p in sorted(Path(SERVER_DIR, "engines").glob("*.py")) if p.name != "__init__.py")
names = [n.strip() for n in INSTALL.split(",") if n.strip()]
assert names and all(n in ENGINE_META for n in names), f"INSTALL inválido; motores: {list(ENGINE_META)}"
assert DEFAULT_ENGINE in names, f"DEFAULT_ENGINE ({DEFAULT_ENGINE}) precisa estar em INSTALL"
gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
assert gpu, ("Sem GPU: Settings > Accelerator > GPU T4 x2, depois rode de novo" if Path("/kaggle").is_dir()
             else "Sem GPU: Runtime > Change runtime type > T4 GPU, depois rode de novo")
print(gpu)

sh(f"pip install -q -r {SERVER_DIR}/requirements.txt uv")
ENGINE_PYTHON = {}
for n in names:
    meta = ENGINE_META[n]
    if meta["venv"]:  # pins that clash with other engines: own venv, its workers run under that Python
        venv = f"/content/venvs/{n}"
        # uv, not `python -m venv`: Colab's Python has no ensurepip. Same interpreter as the server,
        # because the worker talks to it through multiprocessing.
        sh(f"test -x {venv}/bin/python || uv venv -q --python {sys.executable} {venv}")
        sh(f"uv pip install -q --python {venv}/bin/python numpy {meta['pip']}")
        ENGINE_PYTHON[n] = f"{venv}/bin/python"
    else:
        sh(f"pip install -q {meta['pip']}")
    print(f"✓ {n} instalado" + (" (venv próprio)" if meta["venv"] else ""))

if "f5" in names:
    # torchaudio.load (used by F5) needs torchcodec to match the installed torch
    sh(f"{ENGINE_PYTHON.get('f5', sys.executable)} -c \"import torch, torchcodec; print('torch', torch.__version__, '| torchcodec', torchcodec.__version__, '| ok')\"")
    sh("mkdir -p /content/model && (test -s /content/model/model_last.safetensors || wget -q --show-progress -O /content/model/model_last.safetensors https://huggingface.co/firstpixel/F5-TTS-pt-br/resolve/main/pt-br/model_last.safetensors)")
sh("test -x /usr/local/bin/cloudflared || (wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared)")
print("setup ok")
