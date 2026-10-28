"""Builds colab/f5_tts_server.ipynb: a thin launcher. Cell 1 holds the settings and clones/pulls the
repo; cells 2-4 just %run colab/setup.py, start.py and monitor.py from it. Everything that changes
lives in the repo, so the notebook only needs re-uploading if these few lines change.

Run:  python3 scripts/build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "f5_tts_server.ipynb"
REPO = "https://github.com/Hugobsan/f5-tts-colab"

INTRO = """# Servidor de TTS (pt-BR) no Colab ou no Kaggle

Sobe o motor de voz (Qwen3-TTS por padrão; F5-TTS pt-BR e Chatterbox disponíveis para comparar) na GPU e publica a API via Cloudflare Tunnel para o client `f5tts` da sua máquina. O mesmo notebook roda no Colab e no Kaggle (detecta sozinho). O código vem do GitHub a cada execução da célula 1: este notebook não precisa ser trocado quando o projeto muda.

**Antes da primeira vez** (detalhes no README do projeto). Secrets:
- `F5_TOKEN`: o token gerado por `f5tts config --novo-token` na sua máquina.
- `CF_TUNNEL_TOKEN` *(opcional)*: saída de `cloudflared tunnel token f5`. Sem ele, cai no quick tunnel (URL aleatória).

| | Colab | Kaggle |
|---|---|---|
| GPU | *Runtime → Change runtime type → **T4 GPU*** | *Settings → Accelerator → **GPU T4 x2*** (usa as duas) |
| Internet | já vem ligada | *Settings → **Internet on*** (conta com telefone verificado) |
| Secrets | 🔑 na barra lateral, liberar acesso ao notebook | *Add-ons → Secrets*, marcar o checkbox de cada um |
| Rodar | *Runtime → Run all*, aba aberta | *Save Version → **Save & Run All***: roda em segundo plano até 12 h, sem aba aberta. Cancele quando acabar (a célula 4 roda em loop e gasta a cota de 30 h/semana) |

**Um server por vez no mesmo túnel:** a célula 3 se recusa a subir se outro (Colab ou Kaggle) ainda responder na URL. Desligue o outro antes.

**Toda sessão:** rode tudo. A célula 3 imprime a URL; a célula 4 fica mostrando o status.

**Motores:** os de `INSTALL` são instalados; só o `DEFAULT_ENGINE` sobe carregado. Um job com `--modelo qwen3` carrega o motor na hora (~1 min na primeira vez) e ele é descarregado depois de 10 min sem uso.

**Atualizou o projeto no GitHub ou mudou a célula 1:** interrompa a célula 4 e rode de novo as células 1, 2 e 3.
"""

CONFIG = f"""# 1. Configuração + código do projeto (clona ou atualiza do GitHub)
INSTALL = "qwen3"                          # motores instalados ("qwen3,f5,chatterbox" para comparar); os extras carregam sob demanda
DEFAULT_ENGINE = "qwen3"                   # sobe carregado e atende jobs sem --modelo (venceu o comparativo de 2026-10-27)
WORKERS = "auto"                           # workers do motor padrão: "auto" (cabe na RAM/VRAM) ou um número
PUBLIC_URL = "https://f5.hugobsan.com.br"  # hostname do túnel nomeado (usado só com CF_TUNNEL_TOKEN)
REPO, BRANCH = "{REPO}", "main"

import subprocess
r = subprocess.run(f"if [ -d /content/repo/.git ]; then git -C /content/repo fetch -q --depth 1 origin {{BRANCH}} && "
                   f"git -C /content/repo reset -q --hard FETCH_HEAD; else git clone -q --depth 1 -b {{BRANCH}} {{REPO}} /content/repo; fi",
                   shell=True)
assert r.returncode == 0, "git falhou. No Kaggle: Settings > Internet on (precisa de telefone verificado na conta)"
SERVER_DIR = "/content/repo/server"
print("projeto em", subprocess.run(["git", "-C", "/content/repo", "log", "-1", "--format=%h · %s · %cr"],
                                   capture_output=True, text=True).stdout)
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
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True, "language": "python"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        cell("markdown", INTRO),
        cell("code", CONFIG),
        cell("code", "# 2. Dependências e motores (~2-4 min por motor na primeira vez da sessão)\n%run -i /content/repo/colab/setup.py"),
        cell("code", "# 3. Sobe o server + túnel e espera o motor padrão ficar pronto\n%run -i /content/repo/colab/start.py"),
        cell("code", "# 4. Mantém a sessão ativa e mostra o status a cada minuto (interrompa para rodar outra célula)\n%run -i /content/repo/colab/monitor.py"),
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"gerado {OUT.relative_to(ROOT)} (lançador; o código vem de {REPO})")
