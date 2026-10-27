# Notebook cell 3: `%run -i /content/repo/colab/start.py`
# Starts the server (default engine loaded, the others on demand) and the Cloudflare tunnel, then
# waits for the default engine's workers. Uses INSTALL/DEFAULT_ENGINE/PUBLIC_URL (cell 1) and
# ENGINE_PYTHON (setup.py).
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from google.colab import userdata


def secret(name):
    try:
        return userdata.get(name)
    except Exception:  # secret missing or notebook access not granted
        return None


F5_TOKEN = secret("F5_TOKEN")
assert F5_TOKEN, "Crie o Secret F5_TOKEN (🔑 na barra lateral) e libere o acesso ao notebook"
CF_TUNNEL_TOKEN = secret("CF_TUNNEL_TOKEN")

subprocess.run("pkill -f server/run.py; pkill -f multiprocessing.spawn; pkill -f 'cloudflared tunnel'", shell=True)  # re-run
time.sleep(2)
env = dict(os.environ, F5_TOKEN=F5_TOKEN, F5_INSTALLED=INSTALL, F5_DEFAULT_ENGINE=DEFAULT_ENGINE,
           F5_WORKERS=str(globals().get("WORKERS", "auto")), PORT="8000",
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
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/tunnel.log").read())
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
    return " · ".join(f"{n} {e['ready']}/{e['workers']}" for n, e in h["engines"].items()) or "nenhum carregado"


t0 = time.time()
while True:
    assert server.poll() is None, "server morreu:\n" + open("/content/server.log").read()[-3000:]
    try:
        h = health()
        print(f"\r{int(time.time() - t0)}s · prontos: {engines_line(h)} (carregando um por vez)", end="")
        if h["loading"] == 0:
            break
    except OSError:
        pass  # server still starting
    assert time.time() - t0 < 1800, "workers não terminaram de carregar em 30 min: veja /content/server.log"
    time.sleep(3)

if h["broken"]:
    print(f"\n⚠️  workers quebrados (o server segue com os outros): {h['broken']}")
assert h["ready"] > 0, "nenhum worker carregou: veja /content/server.log"
print(f"\n\n✅ No ar: {url} · padrão: {h['default_engine']} · instalados (carregam sob demanda): {', '.join(h['installed'])}")
print(f"Na sua máquina:  f5tts config --url {url}   (uma vez; com túnel nomeado a URL não muda)\n                 f5tts status")
