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

INTRO = """# Servidor de TTS (pt-BR) no Colab

Sobe os motores de voz (F5-TTS pt-BR e outros para comparar) na GPU do Colab e publica a API via Cloudflare Tunnel para o client `f5tts` da sua máquina. O código vem do GitHub a cada execução da célula 1: este notebook não precisa ser trocado quando o projeto muda.

**Antes da primeira vez** (detalhes no README do projeto):
1. *Runtime → Change runtime type → **T4 GPU*** (este notebook já pede GPU; confira).
2. Em 🔑 **Secrets** (barra lateral), crie e libere o acesso deste notebook a:
   - `F5_TOKEN`: o token gerado por `f5tts config --novo-token` na sua máquina.
   - `CF_TUNNEL_TOKEN` *(opcional)*: saída de `cloudflared tunnel token f5`. Sem ele, cai no quick tunnel (URL aleatória).

**Toda sessão:** *Runtime → Run all*. A célula 3 imprime a URL; a célula 4 fica mostrando o status (deixe a aba aberta).

**Motores:** os de `INSTALL` são instalados; só o `DEFAULT_ENGINE` sobe carregado. Um job com `--modelo qwen3` carrega o motor na hora (~1 min na primeira vez) e ele é descarregado depois de 10 min sem uso.

**Atualizou o projeto no GitHub ou mudou a célula 1:** interrompa a célula 4 e rode de novo as células 1, 2 e 3.
"""

CONFIG = f"""# 1. Configuração + código do projeto (clona ou atualiza do GitHub)
INSTALL = "f5,qwen3,chatterbox"            # motores instalados; os que não são o padrão carregam sob demanda
DEFAULT_ENGINE = "f5"                      # sobe carregado e atende jobs sem --modelo
WORKERS = "auto"                           # workers do motor padrão: "auto" (cabe na RAM/VRAM) ou um número
PUBLIC_URL = "https://f5.hugobsan.com.br"  # hostname do túnel nomeado (usado só com CF_TUNNEL_TOKEN)
REPO, BRANCH = "{REPO}", "main"

import subprocess
subprocess.run(f"if [ -d /content/repo/.git ]; then git -C /content/repo fetch -q --depth 1 origin {{BRANCH}} && "
               f"git -C /content/repo reset -q --hard FETCH_HEAD; else git clone -q --depth 1 -b {{BRANCH}} {{REPO}} /content/repo; fi",
               shell=True, check=True)
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
