"""Builds colab/f5_tts_server.ipynb. The notebook clones/pulls the server code from GitHub,
so it only needs regenerating (and re-uploading) when the notebook cells themselves change.

Run:  python3 scripts/build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "f5_tts_server.ipynb"
REPO = "https://github.com/hugobsan/f5-tts-colab"

INTRO = """# Servidor de TTS (pt-BR) no Colab

Sobe os motores de voz (F5-TTS pt-BR e, opcionalmente, outros para comparar) na GPU do Colab e publica a API via Cloudflare Tunnel para o client `f5tts` da sua máquina. O código do server vem do GitHub a cada execução da célula 1, então este notebook não precisa ser trocado quando o server muda.

**Antes da primeira vez** (detalhes no README do projeto):
1. *Runtime → Change runtime type → **T4 GPU*** (este notebook já pede GPU; confira).
2. Em 🔑 **Secrets** (barra lateral), crie e libere o acesso deste notebook a:
   - `F5_TOKEN`: o token gerado por `f5tts config --novo-token` na sua máquina.
   - `CF_TUNNEL_TOKEN` *(opcional)*: saída de `cloudflared tunnel token f5`. Sem ele, cai no quick tunnel (URL aleatória).

**Toda sessão:** ajuste `ENGINES` na célula 1 se quiser outros motores, depois *Runtime → Run all*. A célula 3 imprime a URL. A célula 4 fica rodando e mostrando o status: deixe a aba aberta.

**Server atualizado no GitHub / trocou `ENGINES`:** interrompa a célula 4 e rode de novo as células 1, 2 e 3.
"""

CONFIG = f"""# 1. Configuração + código do server (clona ou atualiza do GitHub)
ENGINES = "f5:auto"                        # motores e nº de workers; o 1º é o padrão. Ex.: "f5:3,qwen3:1,chatterbox:1"
                                           # "auto" (só com 1 motor) = quantos couberem na RAM/VRAM livres (~4 F5 na T4 grátis)
PUBLIC_URL = "https://f5.hugobsan.com.br"  # hostname do túnel nomeado (usado só com CF_TUNNEL_TOKEN)
REPO, BRANCH = "{REPO}", "main"

import subprocess, sys

def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)

sh(f"if [ -d /content/repo/.git ]; then git -C /content/repo fetch -q --depth 1 origin {{BRANCH}} && "
   f"git -C /content/repo reset -q --hard FETCH_HEAD; else git clone -q --depth 1 -b {{BRANCH}} {{REPO}} /content/repo; fi")
SERVER_DIR = "/content/repo/server"
print("server em", subprocess.run(["git", "-C", "/content/repo", "log", "-1", "--format=%h · %s · %cr"],
                                 capture_output=True, text=True).stdout)
"""

SETUP = """# 2. Dependências dos motores escolhidos (~2-4 min por motor na primeira vez da sessão)
import ast
from pathlib import Path

def engine_meta(path):
    # NAME/PIP/VENV constants of an engine module, read without importing it
    consts = {t.id: ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and t.id in ("NAME", "PIP", "VENV")}
    return consts["NAME"], {"pip": consts["PIP"], "venv": consts.get("VENV", False)}

ENGINE_META = dict(engine_meta(p) for p in sorted(Path(SERVER_DIR, "engines").glob("*.py")) if p.name != "__init__.py")
names = [s.split(":")[0].strip() for s in ENGINES.split(",") if s.strip()]
assert names and all(n in ENGINE_META for n in names), f"ENGINES inválido; motores: {list(ENGINE_META)}"
gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
assert gpu, "Sem GPU: Runtime > Change runtime type > T4 GPU, depois rode de novo"
print(gpu)

sh(f"pip install -q -r {SERVER_DIR}/requirements.txt")  # deps do server vêm do repo
ENGINE_PYTHON = {}
for n in names:
    meta = ENGINE_META[n]
    if meta["venv"]:  # pins que brigam com os outros motores: venv próprio, workers rodam no python dele
        venv = f"/content/venvs/{n}"
        sh(f"test -x {venv}/bin/python || python -m venv {venv}")
        sh(f"{venv}/bin/pip install -q numpy {meta['pip']}")
        ENGINE_PYTHON[n] = f"{venv}/bin/python"
    else:
        sh(f"pip install -q {meta['pip']}")
    print(f"✓ {n} instalado" + (" (venv próprio)" if meta["venv"] else ""))

if "f5" in names:
    # torchaudio.load (usado pelo F5) depende do torchcodec casar com o torch instalado
    sh(f"{ENGINE_PYTHON.get('f5', sys.executable)} -c \\"import torch, torchcodec; print('torch', torch.__version__, '| torchcodec', torchcodec.__version__, '| ok')\\"")
    sh("mkdir -p /content/model && (test -s /content/model/model_last.safetensors || wget -q --show-progress -O /content/model/model_last.safetensors https://huggingface.co/firstpixel/F5-TTS-pt-br/resolve/main/pt-br/model_last.safetensors)")
sh("test -x /usr/local/bin/cloudflared || (wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared)")
print("setup ok")
"""

START = """# 3. Sobe o server + túnel e espera os workers ficarem prontos
import json, os, re, time, urllib.request
from google.colab import userdata

def secret(name):
    try:
        return userdata.get(name)
    except Exception:  # secret ausente ou sem acesso liberado
        return None

F5_TOKEN = secret("F5_TOKEN")
assert F5_TOKEN, "Crie o Secret F5_TOKEN (🔑 na barra lateral) e libere o acesso ao notebook"
CF_TUNNEL_TOKEN = secret("CF_TUNNEL_TOKEN")

subprocess.run("pkill -f server/run.py; pkill -f multiprocessing.spawn; pkill -f 'cloudflared tunnel'", shell=True)  # re-execução
time.sleep(2)
env = dict(os.environ, F5_TOKEN=F5_TOKEN, F5_ENGINES=ENGINES, PORT="8000",
           **{f"ENGINE_PYTHON_{n}": py for n, py in ENGINE_PYTHON.items()})
server = subprocess.Popen([sys.executable, f"{SERVER_DIR}/run.py"], cwd=SERVER_DIR, env=env,
                          stdout=open("/content/server.log", "w"), stderr=subprocess.STDOUT)

tunnel_log = open("/content/tunnel.log", "w")
if CF_TUNNEL_TOKEN:
    subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "run", "--url", "http://localhost:8000",
                      "--token", CF_TUNNEL_TOKEN], stdout=tunnel_log, stderr=subprocess.STDOUT)
    url = PUBLIC_URL
else:
    subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://localhost:8000"],
                     stdout=tunnel_log, stderr=subprocess.STDOUT)
    url = None
    for _ in range(60):
        m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open("/content/tunnel.log").read())
        if m:
            url = m.group(0)
            break
        time.sleep(1)
    assert url, "quick tunnel não subiu: veja /content/tunnel.log"

def health():
    req = urllib.request.Request("http://127.0.0.1:8000/health", headers={"x-token": F5_TOKEN})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)

def engines_line(h):
    return " · ".join(f"{n} {e['ready']}/{e['workers']}" for n, e in h["engines"].items())

t0 = time.time()
while True:
    assert server.poll() is None, "server morreu:\\n" + open("/content/server.log").read()[-3000:]
    try:
        h = health()
        print(f"\\r{int(time.time() - t0)}s · prontos: {engines_line(h)} (carregando um por vez)", end="")
        if h["loading"] == 0:
            break
    except OSError:
        pass  # server ainda subindo
    assert time.time() - t0 < 1800, "workers não terminaram de carregar em 30 min: veja /content/server.log"
    time.sleep(3)

if h["broken"]:
    print(f"\\n⚠️  workers quebrados (o server segue com os outros): {h['broken']}")
assert h["ready"] > 0, "nenhum worker carregou: veja /content/server.log"
print(f"\\n\\n✅ No ar: {url} · motor padrão: {h['default_engine']}")
print(f"Na sua máquina:  f5tts config --url {url}   (uma vez; com túnel nomeado a URL não muda)\\n                 f5tts status")
"""

MONITOR = """# 4. Mantém a sessão ativa e mostra o status a cada minuto (interrompa para rodar outra célula; o server continua)
from IPython.display import clear_output

while True:
    clear_output(wait=True)
    try:
        h = health()
        gpu, ram = h["gpu"] or {}, h["ram"]
        print(time.strftime("%H:%M:%S"), f"· {url} · prontos: {engines_line(h)} (vivos {h['alive']})",
              f"· partes na fila {h['pending_parts']}")
        print(f"GPU {gpu.get('util_pct')}% · VRAM {gpu.get('mem_used_mb')}/{gpu.get('mem_total_mb')} MB",
              f"· RAM livre {ram['available_mb']}/{ram['total_mb']} MB · RAM por worker {ram['workers_rss_mb']} MB")
        if h["broken"]:
            print("quebrados:", h["broken"])
        if h["restarts"]:
            print("reinícios:", h["restarts"])
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "server não responde:", e)
    print("\\n--- server.log (fim) ---")
    print("".join(open("/content/server.log").readlines()[-8:]))
    time.sleep(60)
"""


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": [cell("markdown", INTRO), cell("code", CONFIG), cell("code", SETUP), cell("code", START), cell("code", MONITOR)],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"gerado {OUT.relative_to(ROOT)} (código do server vem de {REPO})")
