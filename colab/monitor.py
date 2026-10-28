# Notebook cell 4: `%run -i /content/repo/colab/monitor.py`
# Keeps the session busy and shows the server status every minute. Interrupt it to run another
# cell; the server keeps running. Uses server/health/engines_line/url from start.py.
# When the server exits (`f5tts desligar`), it frees the machine: on Kaggle this cell ends and so does
# the "Save & Run All" run; on Colab it unassigns the runtime.
import os
import subprocess
import time

from IPython.display import clear_output

while server.poll() is None:
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
    print("\n--- server.log (fim) ---")
    print("".join(open("/content/server.log").readlines()[-8:]))
    for _ in range(12):  # 1 min, but notice a shutdown within 5 s
        if server.poll() is not None:
            break
        time.sleep(5)

print(time.strftime("%H:%M:%S"), "server encerrado:\n" + "".join(open("/content/server.log").readlines()[-5:]))
subprocess.run("pkill -f 'cloudflared tunnel'", shell=True)
if not os.path.isdir("/kaggle"):
    from google.colab import runtime
    print("liberando a máquina do Colab")
    runtime.unassign()
