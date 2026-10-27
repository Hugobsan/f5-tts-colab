"""Client check against a fake server (no model needed): python3 tests/test_client.py

Covers: one job per style, the pronunciation dictionary reaching only the text
sent, parts finishing out of order, a transient 502 from the tunnel, a part
that fails validation being regenerated, joining in script order with the
pause, and every job being deleted at the end.
"""

import io
import json
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f5tts import F5Client, load_pronunciation, match_score  # noqa: E402

SR = 1000  # tiny sample rate keeps the math readable
FRAMES = 10


def tone(value):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(value.to_bytes(2, "little", signed=True) * FRAMES)
    return buf.getvalue()


def first_sample(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return int.from_bytes(w.readframes(1), "little", signed=True)


sent = []  # every text the server received; a part's wav carries its index here
jobs = {}  # job_id -> {"ids": [...], "polls": n}
deleted = []
flaky = {"left": 1}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("x-token") != "t":
            return self.reply(401, {"detail": "token"})
        if self.path == "/health":
            return self.reply(200, {"ready": 1, "workers": 1, "loading": 0, "broken": {}})
        parts = self.path.strip("/").split("/")
        job = jobs[parts[1]]
        if len(parts) == 4:
            return self.reply(200, tone(job["ids"][int(parts[3])]), "audio/wav")
        job["polls"] += 1
        if parts[1] == "j1" and job["polls"] == 2 and flaky["left"]:
            flaky["left"] = 0
            return self.reply(502, {"detail": "bad gateway"})
        total = len(job["ids"])
        done = list(range(total))[::-1][:job["polls"]]  # out of order, one more per poll
        return self.reply(200, {"status": "running", "total": total, "done": done, "failed": {}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        job_id = f"j{len(jobs) + 1}"
        jobs[job_id] = {"ids": [], "polls": 0}
        for t in body["texts"]:
            jobs[job_id]["ids"].append(len(sent))
            sent.append(t)
        self.reply(200, {"job_id": job_id, "total": len(body["texts"])})

    def do_DELETE(self):
        deleted.append(self.path.rsplit("/", 1)[1])
        self.reply(200, {})


def check_scoring():
    # digits vs spelled-out numbers, accents and punctuation are not errors
    assert match_score("Em dois mil e vinte e seis, a equipe reduziu.", "Em 2026 a equipe reduziu") == 1.0
    assert match_score("Até lá, e bons códigos!", "ate la e bons codigos") == 1.0
    # dropped words are
    assert match_score("Nos próximos vídeos vamos explorar os recursos.", "Nos próximos vídeos explorar os recursos.") < 0.9


def main():
    check_scoring()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.wav"
        ref.write_bytes(tone(1))
        tsv = Path(tmp) / "p.tsv"
        tsv.write_text("# comentário\nPR\tpi ar\npull request\tpul riquést\n")
        out = Path(tmp) / "out.wav"

        # what whisper "hears": the original term (c) or the respelling itself (d) must both pass
        original = {"a primeira": "a primeira", "b segunda": "b segunda",
                    "c abre o pi ar": "c abre o PR", "d um pul riquést": "d um pul riquest"}
        heard_b = {"n": 0}

        def fake_whisper(wav):
            text = original[sent[first_sample(wav)]]
            if text == "b segunda":
                heard_b["n"] += 1
                if heard_b["n"] == 1:
                    return "algo totalmente diferente"  # first take of b "sounds wrong"
            return text

        c = F5Client(f"http://127.0.0.1:{srv.server_address[1]}", "t", retries=2)
        st = c.synthesize(
            [("", "a primeira"), ("animado", "b segunda"), ("", "   "), ("", "c abre o PR"), ("animado", "d um pull request")],
            out, {"": (ref, "r"), "animado": (ref, "r2")}, pause=0.1, poll=0.01,
            pronunciation=load_pronunciation(tsv), validate=fake_whisper,
        )

        assert sent[:4] == ["a primeira", "c abre o pi ar", "b segunda", "d um pul riquést"], sent  # 1 job per style
        assert sent[4:] == ["b segunda"], sent  # only b regenerated
        assert sorted(deleted) == ["j1", "j2", "j3"], deleted
        assert st["retries"] == 1 and not st["below_min"], st
        assert [p["attempts"] for p in st["parts"]] == [1, 2, 1, 1]

        with wave.open(str(out)) as w:
            samples = [int.from_bytes(w.readframes(1), "little", signed=True) for _ in range(w.getnframes())]
        gap = [0] * 100  # 0.1 s at 1 kHz
        expected = []
        for i, sid in enumerate([0, 4, 1, 3]):  # a, b (2nd take), c, d in script order
            expected += (gap if i else []) + [sid] * FRAMES
        assert samples == expected, "wrong order, take or pause"
    srv.shutdown()
    print("ok")


if __name__ == "__main__":
    main()
