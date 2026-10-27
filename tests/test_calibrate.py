"""Calibrator check with a stub client (no model, no network): python3 tests/test_calibrate.py

Covers: the raw term always being a candidate, all candidates x takes in one job, ranking by how
well the original term is heard (broken carrier last), wavs saved, the job deleted even on failure,
update_tsv replace/append keeping comments, and the `pronuncia` CLI --salvar decision.
"""

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f5tts import calibrate as cal  # noqa: E402
from f5tts.client import F5Error  # noqa: E402

# what the stub Whisper "hears" for each text sent to the model
HEARD = {
    "hoje vamos falar sobre pull request com calma.": "hoje vamos falar sobre pulo reco esto com calma",
    "hoje vamos falar sobre pul riquést com calma.": "hoje vamos falar sobre pull request com calma",
    "hoje vamos falar sobre pôl riquést com calma.": "hoje vamos falar sobre pol request com calma",
    # hears the term perfectly but eats the carrier: must rank last
    "hoje vamos falar sobre púl ri-quést com calma.": "pull request",
}


class Stub:
    def __init__(self, fail_at=None):
        self.jobs, self.deleted, self.polls, self.fail_at = [], [], 0, fail_at

    def ensure_ready(self):
        return {"ready": 1}

    def submit(self, texts, ref_audio, ref_text, nfe=32, speed=1.0, engine=None, opts=None):
        self.jobs.append({"texts": list(texts), "engine": engine, "nfe": nfe})
        return {"job_id": f"j{len(self.jobs)}"}

    def status(self, job_id):
        self.polls += 1
        total = len(self.jobs[-1]["texts"])
        if self.fail_at == self.polls:
            return {"done": [], "failed": {0: "boom"}}
        return {"done": list(range(total))[::-1][:self.polls * 3], "failed": {}}  # out of order, 3 per poll

    def part(self, job_id, n):
        return self.jobs[-1]["texts"][n].encode()

    def delete(self, job_id):
        self.deleted.append(job_id)


def fake_whisper(wav):
    return HEARD[wav.decode()]


def check_term_score():
    assert cal.term_score("pull request", "vamos falar sobre pull request hoje") == 1.0
    assert cal.term_score("pull request", "sobre pullrequest hoje") == 1.0  # glued words still count
    assert cal.term_score("branch", "uma brench nova") > cal.term_score("branch", "uma brincadeira nova")
    assert cal.term_score("branch", "") == 0.0


def check_calibrate():
    stub = Stub()
    with tempfile.TemporaryDirectory() as tmp:
        r = cal.calibrate(stub, "pull request", ["pôl riquést", "púl ri-quést", "pul riquést", "Pull Request"],
                          "ref.wav", "ref", takes=2, engine="e1", transcribe=fake_whisper, save_dir=tmp, poll=0)
        saved = sorted(p.name for p in Path(tmp).iterdir())
    assert len(stub.jobs) == 1, "all candidates x takes must go in one job"
    texts = stub.jobs[0]["texts"]
    assert len(texts) == 8 and texts[0] == texts[1] == "hoje vamos falar sobre pull request com calma.", texts
    assert stub.jobs[0]["engine"] == "e1"
    assert stub.deleted == ["j1"]
    order = [x["spelling"] for x in r]
    assert order == ["pul riquést", "pôl riquést", "pull request", "púl ri-quést"], order  # dup "Pull Request" dropped
    assert r[0]["term"] == 1.0 and not r[0]["broken"]
    assert r[-1]["broken"] and r[-1]["term"] == 1.0 and r[-1]["carrier"] == 0.0
    assert [x["baseline"] for x in r] == [False, False, True, False]
    assert len(r[0]["takes"]) == 2
    assert len(saved) == 8 and "pull-request__00-pull-request__t1.wav" in saved, saved

    failing = Stub(fail_at=1)
    try:
        cal.calibrate(failing, "pull request", ["pul riquést"], "ref.wav", "ref", transcribe=fake_whisper, poll=0)
        raise AssertionError("should have raised")
    except F5Error:
        pass
    assert failing.deleted == ["j1"], "job must be deleted when the run fails"

    for bad in ({"carrier": "sem marcador"}, {"takes": 0}):
        try:
            cal.calibrate(Stub(), "x", [], "r", "r", transcribe=fake_whisper, **bad)
            raise AssertionError(f"should reject {bad}")
        except F5Error:
            pass


def check_update_tsv():
    with tempfile.TemporaryDirectory() as tmp:
        tsv = Path(tmp) / "p.tsv"
        tsv.write_text("# comentário\npull requests\tpul riquésts\n\nPull Request\tvelho\n# fim\n", encoding="utf-8")
        assert cal.update_tsv(tsv, "pull request", "pul riquést") == "replaced"
        assert cal.update_tsv(tsv, "branch", "brén-tchi") == "added"
        assert tsv.read_text(encoding="utf-8") == (
            "# comentário\npull requests\tpul riquésts\n\npull request\tpul riquést\n# fim\nbranch\tbrén-tchi\n")
        new = Path(tmp) / "novo.tsv"
        assert cal.update_tsv(new, "merge", "mêrdj") == "added" and new.read_text() == "merge\tmêrdj\n"


def run_cli(argv, tsv):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    cal.register(sub)
    a = p.parse_args(argv + ["--salvar", str(tsv)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        a.fn(a)
    return out.getvalue()


def check_cli():
    stub = Stub()
    cal.F5Client = lambda: stub
    cal.whisper_transcribe = fake_whisper
    cal.shutil.which = lambda name: "/bin/true"
    cal.time.sleep = lambda s: None
    with tempfile.TemporaryDirectory() as tmp:
        ref, txt = Path(tmp) / "v-ref.wav", Path(tmp) / "v-ref.txt"
        ref.write_bytes(b"x")
        txt.write_text("ref text")
        cal.WHISPER_MODEL = ref  # any existing file
        tsv = Path(tmp) / "p.tsv"
        tsv.write_text("# c\n")
        out = run_cli(["pronuncia", "pull request", "pul riquést", "púl ri-quést", "--voz", str(Path(tmp) / "v")], tsv)
        assert "adicionado" in out and tsv.read_text() == "# c\npull request\tpul riquést\n", out
        assert "⚠" in out and "(cru)" in out, out
        # nothing beats the raw term -> nothing saved
        tsv.write_text("# c\n")
        out = run_cli(["pronuncia", "pull request", "púl ri-quést", "--voz", str(Path(tmp) / "v")], tsv)
        assert "nada salvo" in out and tsv.read_text() == "# c\n", out
        try:
            run_cli(["pronuncia", "pull request", "x", "--frase", "sem marcador", "--voz", str(Path(tmp) / "v")], tsv)
            raise AssertionError("should have raised")
        except F5Error:
            pass


if __name__ == "__main__":
    check_term_score()
    check_calibrate()
    check_update_tsv()
    check_cli()
    print("ok")
