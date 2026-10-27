"""f5tts: command line for the F5-TTS server running on Colab."""

import argparse
import json
import re
import secrets
import shutil
import sys
import time
from pathlib import Path

from . import calibrate
from .client import (CONFIG_PATH, WHISPER_MODEL, F5Client, F5Error, ServerDown, load_config,
                     load_pronunciation, save_config, server_asr, wav_duration, whisper_transcribe)

STYLE_TAG = re.compile(r"^\[([\w-]+)\]\s*(.*)$")


def read_arg(value):
    """A path to an existing file is read; anything else is the literal text."""
    p = Path(value)
    return p.read_text(encoding="utf-8") if p.is_file() else value


def parse_script(text):
    """One part per non-empty line; a leading `[estilo]` picks that style's reference."""
    parts = []
    for line in text.splitlines():
        m = STYLE_TAG.match(line.strip())
        parts.append((m.group(1), m.group(2)) if m else ("", line))
    return parts


def resolve_refs(a, styles):
    """{style: (wav, text)}: --ref/--ref-text is the default style; --voz PREFIX maps style X to PREFIX-X-ref.wav."""
    if a.ref:
        if not a.ref_text:
            raise F5Error("--ref exige --ref-text")
        refs = {"": (a.ref, read_arg(a.ref_text))}
    elif a.voz:
        refs = {}
        for style in styles:
            base = Path(f"{a.voz}-{style}-ref" if style else f"{a.voz}-ref")
            wav, txt = base.with_suffix(".wav"), base.with_suffix(".txt")
            if not (wav.is_file() and txt.is_file()):
                avail = sorted(p.name for p in Path(a.voz).parent.glob(f"{Path(a.voz).name}*-ref.wav"))
                raise F5Error(f"sem ref para o estilo '{style or 'padrão'}': {wav} (existentes: {avail})")
            refs[style] = (str(wav), txt.read_text(encoding="utf-8"))
    else:
        raise F5Error("informe --voz PREFIXO ou --ref + --ref-text")
    return refs


def parse_opts(items):
    """['exaggeration=0.7', 'x=abc'] -> {'exaggeration': 0.7, 'x': 'abc'}"""
    opts = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep:
            raise F5Error(f"--opcao espera CHAVE=VALOR, veio '{item}'")
        try:
            opts[key] = float(value) if "." in value or value.lstrip("-").isdigit() else value
        except ValueError:
            opts[key] = value
    return opts


def pick_asr(client, what):
    """Prints which Whisper will check the parts: the server's (GPU) when ready, else the local whisper-cli."""
    asr = server_asr(client.health())
    if asr:
        print(f"conferência: Whisper do servidor ({asr['model']}, {asr['device']})")
    elif shutil.which("whisper-cli") and WHISPER_MODEL.is_file():
        print("conferência: Whisper local (whisper-cli, CPU)")
    else:
        raise F5Error(f"{what} precisa do Whisper do servidor (F5_ASR_MODEL) ou do whisper-cli no PATH"
                      f" e do modelo {WHISPER_MODEL}")


def progress(st):
    extra = f" · {st['retries']} regeradas" if st["retries"] else ""
    print(f"\r  {st['done']}/{st['total']} partes{extra}  ", end="", file=sys.stderr, flush=True)


def cmd_config(a):
    token = secrets.token_urlsafe(24) if a.novo_token else a.token
    if a.url or token:
        save_config(url=a.url, token=token)
    url, tok = load_config()
    print(f"arquivo: {CONFIG_PATH}\nurl:     {url}\ntoken:   {(tok[:4] + '…' + tok[-4:]) if tok else None}")
    if a.novo_token:
        print(f"\nnovo token (cole no Secret F5_TOKEN do Colab):\n{token}")


def cmd_status(a):
    h = F5Client().health()
    print(json.dumps(h, indent=2, ensure_ascii=False))
    if h["ready"] == 0:
        sys.exit(1)


def cmd_gerar(a):
    parts = parse_script(read_arg(a.entrada))
    refs = resolve_refs(a, {style for style, text in parts if text.strip()})
    client = F5Client()
    validate = None
    if a.validar:
        pick_asr(client, "--validar")
        validate = whisper_transcribe  # fallback only: synthesize prefers the server's ASR
    st = client.synthesize(
        parts, a.saida, refs, nfe=a.nfe, speed=a.speed, pause=a.pausa, sample_rate=a.sr,
        pronunciation=load_pronunciation(a.pronuncia) if a.pronuncia else None,
        validate=validate, min_score=a.min_acerto, max_attempts=a.tentativas,
        engine=a.modelo, opts=parse_opts(a.opcao), on_progress=None if a.quieto else progress,
    )
    if not a.quieto:
        print(file=sys.stderr)
    print(f"[{a.modelo or 'padrão do server'}] gerado em {st['wall_secs']}s ({st['total']} partes, {st['retries']} regeradas) · "
          f"duração {wav_duration(a.saida):.2f}s → {Path(a.saida).resolve()}")
    for r in st["below_min"]:
        print(f"  ⚠ parte {r['n'] + 1} ficou em {r['score']:.0%} após {r['attempts']} tentativas"
              f"\n    texto:  {r['text']}\n    ouvido: {r['heard']}")
    if a.relatorio:
        Path(a.relatorio).write_text(json.dumps(st, indent=2, ensure_ascii=False))


def cmd_bench(a):
    c = F5Client()
    h = c.ensure_ready(a.modelo)
    print(f"[{a.modelo or 'padrão do server'}] workers prontos: {h['ready']}/{h['workers']} · {h['devices']}")
    ref_wav, ref_text = resolve_refs(a, {""})[""]
    t0 = time.time()
    job_id = c.submit([a.frase] * a.partes, ref_wav, ref_text, a.nfe, engine=a.modelo, opts=parse_opts(a.opcao))["job_id"]
    peak = 0
    try:
        while True:
            st = c.status(job_id)
            gpu = c.health()["gpu"]
            peak = max(peak, gpu["util_pct"] if gpu else 0)
            if st["failed"]:
                raise F5Error(f"partes falharam: {st['failed']}")
            if len(st["done"]) == st["total"]:
                break
            time.sleep(1)
    finally:
        c.delete(job_id)
    wall = time.time() - t0
    print(f"{a.partes} partes em {wall:.1f}s · áudio {st['audio_secs']:.1f}s · "
          f"{st['audio_secs'] / wall:.2f}s de áudio por segundo · pico de GPU {peak}%")


def main():
    p = argparse.ArgumentParser(prog="f5tts", description="Client do server F5-TTS no Colab.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("config", help="define/mostra URL e token (~/.config/f5tts/config.json)")
    s.add_argument("--url")
    s.add_argument("--token")
    s.add_argument("--novo-token", action="store_true", help="gera um token novo, salva e imprime")
    s.set_defaults(fn=cmd_config)

    s = sub.add_parser("status", help="checa se o server está no ar (exit 1 se não houver worker pronto)")
    s.set_defaults(fn=cmd_status)

    voice = argparse.ArgumentParser(add_help=False)
    voice.add_argument("--voz", help="prefixo dos refs: PREFIXO-ref.wav/.txt (padrão) e PREFIXO-<estilo>-ref.wav/.txt")
    voice.add_argument("--ref", help="wav de referência único (5-12 s), no lugar de --voz")
    voice.add_argument("--ref-text", help="transcrição exata do --ref (texto ou arquivo .txt)")
    voice.add_argument("--nfe", type=int, default=32, help="passadas (F5): 32 = final, 16 = rascunho")
    voice.add_argument("--modelo", help="motor do server: f5, qwen3, chatterbox (padrão: o primeiro do server)")
    voice.add_argument("--opcao", action="append", metavar="CHAVE=VALOR",
                       help="opção específica do motor, repetível (ex.: --opcao exaggeration=0.7)")

    s = sub.add_parser("gerar", parents=[voice], help="gera a narração; cada linha não vazia vira uma parte")
    s.add_argument("entrada", help="texto ou arquivo .txt (uma parte por linha; '[estilo] texto' escolhe o ref)")
    s.add_argument("saida", help="wav de saída")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--pausa", type=float, default=0.25, help="silêncio entre partes, em segundos")
    s.add_argument("--sr", type=int, help="reamostra a saída (ex.: 44100, usa ffmpeg)")
    s.add_argument("--pronuncia", help="TSV termo<TAB>como falar, aplicado só ao texto enviado ao modelo")
    s.add_argument("--validar", action="store_true", help="confere cada parte com o Whisper (do server, se houver; senão o local) e regera as ruins")
    s.add_argument("--min-acerto", type=float, default=0.9, help="fração mínima de palavras ouvidas (com --validar)")
    s.add_argument("--tentativas", type=int, default=3, help="máximo de gerações por parte (com --validar)")
    s.add_argument("--relatorio", help="salva o relatório por parte (JSON)")
    s.add_argument("--quieto", action="store_true", help="sem barra de progresso")
    s.set_defaults(fn=cmd_gerar)

    calibrate.register(sub)

    s = sub.add_parser("bench", parents=[voice], help="mede a vazão do server com N partes iguais")
    s.add_argument("--partes", type=int, default=6)
    s.add_argument("--frase", default="esta é uma frase de teste para medir quanto tempo o servidor leva para gerar a voz.")
    s.set_defaults(fn=cmd_bench)

    a = p.parse_args()
    try:
        a.fn(a)
    except ServerDown as e:
        sys.exit(f"SERVER FORA DO AR: {e}")
    except F5Error as e:
        sys.exit(f"ERRO: {e}")


if __name__ == "__main__":
    main()
