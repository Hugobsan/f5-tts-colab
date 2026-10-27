# Notebook cell 4: `%run -i /content/repo/colab/monitor.py`
# Keeps the session busy and shows the server status every minute. Interrupt it to run another
# cell; the server keeps running. Uses health/engines_line/url from start.py.
import time

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
    print("\n--- server.log (fim) ---")
    print("".join(open("/content/server.log").readlines()[-8:]))
    time.sleep(60)
