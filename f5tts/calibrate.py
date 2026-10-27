"""Pronunciation calibrator: picks the respelling of an English term that Whisper hears as the original.

Each candidate respelling is spoken inside a carrier sentence, transcribed with the local Whisper and
scored by how well the ORIGINAL term shows up in what was heard. The raw term is always a candidate,
so a respelling only wins if it beats letting the model read the term as is.
"""

import difflib
import shutil
import statistics
import time
from pathlib import Path

from .client import WHISPER_MODEL, F5Client, F5Error, match_score, whisper_transcribe, words

CARRIER = "hoje vamos falar sobre {} com calma."
BROKEN_CARRIER = 0.75  # carrier score below this = the take garbled the sentence, rank the candidate last


def term_score(term, heard):
    """Best fuzzy match (0-1) of the term against any window of the heard words.

    Windows of n-1..n+1 words, compared with spaces removed, so "pullrequest" or "pull re quest"
    still count for "pull request".
    """
    t, h = words(term), words(heard)
    if not t or not h:
        return 0.0
    target = "".join(t)
    best = 0.0
    for size in range(max(1, len(t) - 1), len(t) + 2):
        for i in range(max(1, len(h) - size + 1)):
            best = max(best, difflib.SequenceMatcher(None, target, "".join(h[i:i + size])).ratio())
    return best


def _slug(text):
    return "-".join(words(text)) or "x"


def calibrate(client, term, candidates, ref_wav, ref_text, *, carrier=CARRIER, takes=2, engine=None, nfe=32,
              transcribe=whisper_transcribe, save_dir=None, poll=2.0, stall_timeout=900):
    """Ranks respellings of `term` (best first). Returns [{spelling, baseline, term, carrier, broken, takes}]."""
    if carrier.count("{}") != 1:
        raise F5Error("a frase precisa ter exatamente um {} no lugar do termo")
    if takes < 1:
        raise F5Error("--tentativas precisa ser >= 1")
    cands, seen = [], set()
    for c in [term, *candidates]:
        c = c.strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            cands.append(c)
    texts = [carrier.format(c) for c in cands for _ in range(takes)]  # part n -> cands[n // takes]
    carrier_text = carrier.format("")
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    client.ensure_ready()
    job_id = client.submit(texts, ref_wav, ref_text, nfe=nfe, engine=engine)["job_id"]
    results = {}  # n -> (term score, carrier score, heard)
    try:
        last_change = time.time()
        while len(results) < len(texts):
            st = client.status(job_id)
            if st["failed"]:
                raise F5Error(f"partes falharam no server: {st['failed']}")
            for n in st["done"]:
                if n in results:
                    continue
                wav = client.part(job_id, n)
                ci, take = divmod(n, takes)
                if save_dir:
                    (save_dir / f"{_slug(term)}__{ci:02d}-{_slug(cands[ci])}__t{take + 1}.wav").write_bytes(wav)
                heard = transcribe(wav)
                results[n] = (term_score(term, heard), match_score(carrier_text, heard), heard)
                last_change = time.time()
            if time.time() - last_change > stall_timeout:
                raise F5Error(f"sem progresso há {stall_timeout}s (worker morto? veja `f5tts status`)")
            if len(results) < len(texts):
                time.sleep(poll)
    finally:
        try:
            client.delete(job_id)
        except F5Error:
            pass

    ranking = []
    for ci, c in enumerate(cands):
        rs = [results[ci * takes + k] for k in range(takes)]
        carrier_min = min(r[1] for r in rs)
        ranking.append({"spelling": c, "baseline": ci == 0, "term": round(statistics.mean(r[0] for r in rs), 3),
                        "carrier": round(carrier_min, 3), "broken": carrier_min < BROKEN_CARRIER,
                        "takes": [r[2] for r in rs]})
    # stable sort: on a tie the baseline (listed first) stays ahead, so a respelling must strictly win
    ranking.sort(key=lambda r: (not r["broken"], r["term"], r["carrier"]), reverse=True)
    return ranking


def update_tsv(path, term, spelling):
    """Sets `term<TAB>spelling` in the TSV: replaces the term's line (case-insensitive) or appends it.

    Returns "replaced" or "added". Comments, blank lines and other entries are kept as they are.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    new = f"{term}\t{spelling}"
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("#") and "\t" in line and line.split("\t", 1)[0].strip().lower() == term.lower():
            lines[i] = new
            action = "replaced"
            break
    else:
        lines.append(new)
        action = "added"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return action


def cmd_pronuncia(a):
    from .cli import resolve_refs  # lazy: cli.py imports this module to call register()

    if a.tentativas < 1:
        raise F5Error("--tentativas precisa ser >= 1")
    if not (shutil.which("whisper-cli") and WHISPER_MODEL.is_file()):
        raise F5Error(f"pronuncia precisa do whisper-cli no PATH e do modelo {WHISPER_MODEL}")
    ref_wav, ref_text = resolve_refs(a, {""})[""]
    ranking = calibrate(F5Client(), a.termo, a.candidatos, ref_wav, ref_text, carrier=a.frase,
                        takes=a.tentativas, engine=a.modelo, nfe=a.nfe, transcribe=whisper_transcribe,
                        save_dir=a.saida_dir)

    width = max(len(r["spelling"]) for r in ranking) + 8
    print(f"{'grafia':<{width}} termo  frase  ouvido")
    for r in ranking:
        label = r["spelling"] + (" (cru)" if r["baseline"] else "") + (" ⚠" if r["broken"] else "")
        print(f"{label:<{width}} {r['term']:>5.0%}  {r['carrier']:>5.0%}  {' | '.join(r['takes'])}")
    if any(r["broken"] for r in ranking):
        print(f"⚠ = a frase saiu quebrada em alguma tentativa (frase < {BROKEN_CARRIER:.0%}); vai para o fim")
    if a.saida_dir:
        print(f"wavs em {Path(a.saida_dir).resolve()}")

    if a.salvar:
        best, base = ranking[0], next(r for r in ranking if r["baseline"])
        if best["baseline"] or best["broken"] or best["term"] <= base["term"]:
            print(f"nada salvo em {a.salvar}: nenhuma grafia superou o termo cru ({base['term']:.0%})")
        else:
            action = update_tsv(a.salvar, a.termo, best["spelling"])
            verb = "substituído" if action == "replaced" else "adicionado"
            print(f"{verb} em {a.salvar}: {a.termo}\t{best['spelling']} ({best['term']:.0%} × cru {base['term']:.0%})")


def register(subparsers):
    s = subparsers.add_parser("pronuncia", help="calibra a grafia de um termo em inglês com o Whisper")
    s.add_argument("termo", help='termo original, ex.: "pull request"')
    s.add_argument("candidatos", nargs="*", help="grafias a testar (o termo cru entra sempre como referência)")
    s.add_argument("--voz", help="prefixo dos refs: usa PREFIXO-ref.wav/.txt")
    s.add_argument("--ref", help="wav de referência, no lugar de --voz")
    s.add_argument("--ref-text", help="transcrição exata do --ref (texto ou arquivo .txt)")
    s.add_argument("--modelo", help="engine do server (padrão: o do server)")
    s.add_argument("--tentativas", type=int, default=2, help="gerações por grafia (seeds diferentes)")
    s.add_argument("--frase", default=CARRIER, help="frase portadora com {} no lugar do termo")
    s.add_argument("--nfe", type=int, default=32)
    s.add_argument("--saida-dir", help="salva cada wav aqui para ouvir")
    s.add_argument("--salvar", help="TSV de pronúncia: grava a vencedora se ela superar o termo cru")
    s.set_defaults(fn=cmd_pronuncia)
