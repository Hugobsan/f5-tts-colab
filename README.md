# f5-tts-colab

Narração pt-BR com voz clonada gerada na **GPU do Google Colab** e pedida da sua máquina por um client de linha de comando. O modelo padrão é o **Qwen3-TTS**, e o F5-TTS pt-BR e o Chatterbox ficam disponíveis para comparar.

```
sua máquina                                    Google Colab (T4)
───────────                                    ─────────────────
agente de vídeo ─► f5tts gerar ─HTTPS─► f5.hugobsan.com.br ─► cloudflared ─► f5_server.py
                    (client)    (Cloudflare Tunnel)                          ├─ fila de partes por motor
                    ├─ estilos por linha                                     ├─ supervisor (sob demanda)
                    ├─ dicionário de pronúncia                               ├─ workers do motor (GPU)
                    └─ conferência (Whisper do server)                       └─ worker do Whisper (GPU)
```

**Qual modelo (comparativo de 2026-10-27, mesmas 8 frases e mesmas referências do Theo):**

| | F5 pt-BR | **Qwen3-TTS 1.7B** | Chatterbox pt-BR |
|---|---|---|---|
| acerto do Whisper | 97% | 98% | 97% |
| avaliação de ouvido (👍 +1 / 😐 0 / 👎 −1) | +1 | **+7** | +3 |
| termos em inglês ("pull request") | aproximados, precisa de dicionário | corretos, sem ajuda | corretos |
| sotaque | neutro | neutro; raramente puxa pro nordestino no "d" | carioca |
| licença | não comercial | Apache 2.0 | MIT |

Por isso o notebook sobe com `INSTALL = "qwen3"` e `DEFAULT_ENGINE = "qwen3"`. Para comparar de novo, use `INSTALL = "qwen3,f5,chatterbox"`: os extras carregam sob demanda.

- **Server** (`server/f5_server.py`): FastAPI com fila de jobs. Cada linha do roteiro vira uma *parte*, e N processos worker (uma cópia do modelo cada) geram as partes em paralelo. Um supervisor sobe os workers um por vez e recria os que morrem.
- **Client** (`f5tts/`): CLI + lib Python, só com stdlib. Checa se o server está no ar, envia os jobs, acompanha por polling, baixa as partes, confere cada uma com o Whisper, regera as ruins, junta na ordem e salva o `.wav`.
- **Sem limite de 100 s do Cloudflare.** Toda requisição responde em milissegundos (enviar, consultar, baixar). A geração acontece em segundo plano no Colab.
- **Server sem estado de voz.** Cada job leva o wav de referência e a transcrição, então a voz fica na sua máquina, e o Colab pode ser descartado a qualquer hora.

## Estrutura

| caminho | o que é |
|---|---|
| `server/run.py` | ponto de entrada (`python run.py`) |
| `server/f5_server.py` | API, fila, supervisor dos workers |
| `server/worker.py` | processo worker: carrega um motor e gera as partes da fila dele |
| `server/engines/` | um arquivo por motor de TTS (`f5`, `qwen3`, `chatterbox`), todos com o mesmo contrato (veja `engines/__init__.py`) |
| `colab/f5_tts_server.ipynb` | notebook para subir no Colab (busca o `server/` deste repositório no GitHub) |
| `scripts/build_notebook.py` | regera o notebook quando as células mudam |
| `f5tts/` | client: `client.py` (lib), `cli.py` (comando `f5tts`), `calibrate.py` (calibrador de pronúncia) |
| `tests/` | testes do client e do calibrador contra um server falso (sem modelo) |

---

## Configuração (uma vez só)

### 1. Instale o client na sua máquina

```bash
uv tool install -e ~/codes/f5-tts-colab     # instala o comando `f5tts` em ~/.local/bin
f5tts --help
```

### 2. Gere o token da API e aponte para o túnel

```bash
f5tts config --url https://f5.hugobsan.com.br --novo-token
```

O comando grava `~/.config/f5tts/config.json` (permissão 600) e imprime o token **uma vez**. Copie-o para o passo 4. Para ver a config depois: `f5tts config`. Também dá para usar `F5_URL`/`F5_TOKEN` no ambiente, que têm prioridade sobre o arquivo.

### 3. Pegue o token do túnel nomeado

O túnel `f5` (hostname `f5.hugobsan.com.br`) já foi criado com `tunnel new --alias=f5 --port=8000`. O Colab não precisa de login no Cloudflare, só do token desse túnel:

```bash
cloudflared tunnel token f5
```

> ⚠️ Não rode `tunnel f5` na sua máquina enquanto o Colab estiver no ar: os dois viram conectores do mesmo túnel e o Cloudflare reparte as requisições entre eles.

### 4. Importe o notebook no Colab e crie os Secrets

1. Abra <https://colab.research.google.com> → **File → Upload notebook** → escolha `colab/f5_tts_server.ipynb`.
   (Ele fica salvo no seu Google Drive em *Colab Notebooks*. Nas próximas vezes, abra de lá.)
2. Confira **Runtime → Change runtime type → T4 GPU**. O notebook já pede GPU, mas o plano grátis pode não ter uma disponível na hora.
3. Clique no 🔑 **Secrets** na barra lateral esquerda e crie:

   | nome | valor | obrigatório |
   |---|---|---|
   | `F5_TOKEN` | o token do passo 2 | sim |
   | `CF_TUNNEL_TOKEN` | a saída do passo 3 | não (sem ele, usa quick tunnel) |

   Ligue o **Notebook access** de cada um. Os Secrets ficam na sua conta e valem para as próximas sessões, mas o acesso é liberado **por notebook**.

### Atualizar o server no Colab

O notebook é só um **lançador**: a célula 1 tem as configurações e clona este repositório (<https://github.com/Hugobsan/f5-tts-colab>, público) ou faz pull dele, e as células 2-4 rodam `colab/setup.py`, `colab/start.py` e `colab/monitor.py` do repo. Para atualizar qualquer coisa (server, motores, instalação):
1. Faça commit e `git push`.
2. No Colab, interrompa a célula 4 e rode de novo as células **1, 2 e 3**. A 1 puxa o código novo e mostra o commit em uso, e a 3 reinicia o server.

Só é preciso subir o notebook de novo se o próprio lançador mudar (`scripts/build_notebook.py`). Nesse caso: **Runtime → Disconnect and delete runtime** no antigo, **File → Upload notebook** com o novo, ligar o **Notebook access** dos Secrets e **Run all**.

O client não precisa de nada: ele foi instalado em modo editável (`-e`), então já roda o código atual.

### Rodar no Kaggle em vez do Colab

O mesmo notebook roda no Kaggle e detecta sozinho onde está. Vale para sessões longas: o Kaggle dá até 12 h por sessão (30 h de GPU por semana), duas T4 de 16 GB e ~29 GB de RAM. Lá o `WORKERS = "auto"` divide os workers entre as duas GPUs, com o Whisper na GPU 0.

1. *Create → New Notebook → File → Import Notebook* com `colab/f5_tts_server.ipynb`.
2. *Settings → Accelerator → **GPU T4 x2*** e *Settings → **Internet on***. A internet exige conta com telefone verificado.
3. *Add-ons → Secrets*: crie `F5_TOKEN` e `CF_TUNNEL_TOKEN` com os mesmos valores do Colab e marque o checkbox de cada um neste notebook.
4. *Save Version → **Save & Run All***. Roda em segundo plano, sem aba aberta, e os logs aparecem na página da versão. Quando terminar, **cancele a execução**, porque a célula 4 roda em loop e gasta a cota até o limite de 12 h.

**Um server por vez no mesmo túnel.** Com dois ligados, o Cloudflare reparte as requisições entre eles: um job enviado a um é consultado no outro, que responde 404. Por isso a célula 3 se recusa a subir se outro server ainda responder em `PUBLIC_URL`. Para trocar de plataforma: espere a narração em andamento acabar, desligue um e ligue o outro. A URL continua a mesma.

---

## Toda sessão

1. Abra o notebook no Colab → **Runtime → Run all**.
   - **Célula 1:** configurações e clone ou pull do projeto em `/content/repo/`:
     - `INSTALL = "f5,qwen3,chatterbox"`: motores instalados. Tire os que não for usar para a sessão subir mais rápido.
     - `DEFAULT_ENGINE = "f5"`: o motor que sobe carregado e atende jobs sem `--modelo`.
     - `WORKERS = "auto"`: quantos workers do motor padrão.
   - **Célula 2:** checa a GPU e instala as dependências do server (inclusive o faster-whisper) e os motores de `INSTALL`, cada um no ambiente principal ou num venv próprio, se as dependências dele brigam com as dos outros. Com o F5, baixa o checkpoint pt-BR (~1,3 GB). Leva ~2-4 min por motor.
   - **Célula 3:** sobe o server e o túnel e espera os workers do motor padrão e o Whisper carregarem, **um de cada vez**, e passarem no warmup. No fim imprime `✅ No ar: https://f5.hugobsan.com.br`, e avisa se algum worker quebrou.
   - **Célula 4:** fica em loop mostrando workers, fila, GPU, RAM, quebras e reinícios a cada minuto. **Deixe a aba aberta**, porque o Colab derruba sessões ociosas.

**Motores sob demanda:** um job com `--modelo qwen3` (ou outro de `INSTALL`) faz o server liberar workers do motor padrão até caber na RAM e carregar o motor pedido. A primeira parte demora ~1 min a mais, porque inclui baixar os pesos na primeira vez, e o client mostra "carregando qwen3". Depois de 10 min sem uso (`F5_IDLE_UNLOAD`), ele é descarregado e o motor padrão volta ao tamanho original. Se o motor não carregar (falta de RAM, por exemplo), os jobs dele falham com o motivo, e o padrão segue funcionando.
2. Na sua máquina:
   ```bash
   f5tts status      # exit 0 = pronto; mostra workers, RAM, GPU, quebrados e reinícios
   ```

Sem `CF_TUNNEL_TOKEN`, a célula 3 imprime uma URL `*.trycloudflare.com` que muda a cada sessão. Rode `f5tts config --url <essa-url>` toda vez.

Para desligar, use **Runtime → Disconnect and delete runtime**. Isso também libera a sua cota de GPU.

---

## Uso

### Roteiro

Cada linha não vazia é uma *parte*, gerada de forma independente e em paralelo. As partes são unidas na ordem com `--pausa` de silêncio entre elas.

```text
[animado] Olá! Neste vídeo vamos aprender a usar o painel do tech lead.
Para começar, abra o terminal na pasta do projeto e rode o comando de instalação.
[pergunta] Você sabia que dá para revisar dez pull requests ao mesmo tempo?
[alerta] Atenção: antes de aprovar uma mudança, confira os achados críticos.
```

- **Tom:** o F5 copia o jeito de falar da referência (ritmo, energia, entonação). Não existe tag de emoção no modelo. `[estilo]` no começo da linha troca a referência daquela parte: com `--voz PREFIXO`, a linha sem tag usa `PREFIXO-ref.wav/.txt` e `[animado]` usa `PREFIXO-animado-ref.wav/.txt`. As referências do Theo estão em `~/.claude/skills/f5-tts-voz/refs/`.
- **Tamanho:** linhas de 1-3 frases funcionam bem. Linhas muito longas viram uma parte lenta que segura o job, e frases soltas demais perdem a entonação entre elas.
- **Números por extenso.** O texto vai para minúsculas no server, porque o checkpoint pt-BR foi treinado assim.

### CLI

```bash
f5tts gerar roteiro.txt saida.wav \
  --voz ~/.claude/skills/f5-tts-voz/refs/theo \
  --pronuncia ~/.claude/skills/f5-tts-colab/pronuncia.tsv \
  --validar --sr 44100
```

| opção | padrão | |
|---|---|---|
| `--voz PREFIXO` | | refs por estilo (`PREFIXO-ref.*`, `PREFIXO-<estilo>-ref.*`) |
| `--ref` + `--ref-text` | | uma referência só, no lugar de `--voz` (sem estilos) |
| `--nfe` | 32 | passadas de difusão: 32 = final, 16 = rascunho (~2× mais rápido) |
| `--speed` | 1.0 | velocidade da fala (0,5-2,0) |
| `--pausa` | 0.25 | segundos de silêncio entre partes |
| `--sr` | 24000 | taxa de amostragem da saída (ffmpeg) |
| `--pronuncia` | | TSV `termo<TAB>como falar` |
| `--validar` | | confere cada parte com o Whisper local e regera as ruins |
| `--min-acerto` | 0.9 | fração mínima de palavras ouvidas (com `--validar`) |
| `--tentativas` | 3 | máximo de gerações por parte (com `--validar`) |
| `--relatorio` | | salva o relatório por parte em JSON |
| `--modelo` | padrão do server | motor: `f5`, `qwen3`, `chatterbox` (qualquer um de `INSTALL`; os que não são o padrão carregam sob demanda) |
| `--opcao CHAVE=VALOR` | | opção específica do motor, repetível (ex.: `--opcao exaggeration=0.7`) |
| `--quieto` | | sem barra de progresso |

`entrada` e `--ref-text` aceitam um caminho de arquivo ou o texto direto.

**Códigos de saída:** `0` ok · `1` erro (mensagem começando com `SERVER FORA DO AR:` ou `ERRO:`).

### Pronúncia de termos em inglês

O checkpoint pt-BR só fala português: ele lê "pull request" com regras de leitura do português. O `--pronuncia` aplica um TSV de substituições **só no texto enviado ao modelo**. A tela e a validação continuam usando o termo original:

```text
pull request	pul riquést
deploy	diplói
```

Sem diferenciar maiúsculas, por palavra inteira, e os termos mais longos têm prioridade. A grafia se acerta de ouvido: gere, ouça, ajuste e gere de novo. Com `--validar`, a nota da parte também mostra se o Whisper entendeu o termo como o original.

### Calibrador de pronúncia (`f5tts pronuncia`)

Escolhe a grafia de um termo com dados em vez de chute. Cada candidata (e o termo cru, como referência) é gerada numa frase de teste, num job só, com `--tentativas` seeds cada. O Whisper transcreve, e as candidatas são ordenadas pelo quanto o Whisper ouviu o termo **original**. Uma frase que saiu quebrada é marcada com ⚠ e vai para o fim.

```bash
f5tts pronuncia "pull request" "pul riquést" "púl ri-quést" "pul requést" \
  --voz ~/.claude/skills/f5-tts-voz/refs/theo --tentativas 3 \
  [--modelo f5] [--frase "hoje vamos falar sobre {} com calma."] \
  [--saida-dir DIR]          # guarda os wavs para ouvir
  [--salvar pronuncia.tsv]   # grava a vencedora, só se ela bater o termo cru
```

A nota compara letras, não som, e o Whisper tende a "corrigir" para a palavra em inglês que conhece. Ela serve para ordenar candidatas, não como medida absoluta. Com 3 ou mais tentativas o ranking fica mais estável. A skill `f5-tts-colab` manda o agente que escreve os roteiros calibrar cada termo novo antes de gerar.

### Conferência automática (`--validar`)

O F5 às vezes escorrega: come uma palavra, emenda duas ou corta o fim. No teste, a mesma frase gerada 4 vezes saiu certa 2 vezes e errada 2. Com `--validar`, cada parte é transcrita e comparada com o texto original. A transcrição roda **no server, na GPU** (faster-whisper `small`, worker próprio) quando o `/health` mostra o ASR pronto: o client recebe o texto junto com a parte, e o próprio comando avisa `conferência: Whisper do servidor`. Se o server não tiver ASR (`F5_ASR_MODEL=off` ou server antigo), ela cai no `whisper-cli` local (modelo `ggml-small`, o mesmo do HyperFrames). A nota é a fração das palavras esperadas encontradas, em ordem, desconsiderando acentos, pontuação e números (o Whisper escreve "2026" onde o roteiro diz "dois mil e vinte e seis"). Abaixo de `--min-acerto`, a parte é regerada com outra seed, até `--tentativas`, e fica a melhor. As que terminam abaixo do mínimo aparecem com `⚠`, com o texto e o que foi ouvido.

Limites conhecidos: o próprio Whisper erra um pouco ("abre" × "abra"), e uma palavra perdida em 20 ainda passa nos 90%.

### Lib Python

```python
from f5tts import F5Client, load_pronunciation, whisper_transcribe

refs = {"": ("theo-ref.wav", "transcrição do ref neutro"),
        "animado": ("theo-animado-ref.wav", "transcrição do ref animado")}
st = F5Client().synthesize(                  # F5Client lê ~/.config/f5tts/config.json (ou url=, token=)
    [("animado", "Olá, pessoal!"), ("", "Hoje vamos ver o painel.")], "saida.wav", refs,
    nfe=32, sample_rate=44100, pronunciation=load_pronunciation("pronuncia.tsv"),
    validate=whisper_transcribe,              # ou qualquer função wav_bytes -> texto ouvido
)
print(st["retries"], st["below_min"])        # relatório por parte em st["parts"]
```

Também há métodos de baixo nível: `health()`, `ensure_ready()`, `submit()`, `status()`, `part()` e `delete()`.

---

## Quantos workers usar

Cada worker é uma cópia do modelo e processa uma parte por vez. Medido no Colab grátis (T4, 15 GB de VRAM, ~12,7 GB de RAM):

- **VRAM:** ~0,9 GB por worker parado (4 workers = 3,6 GB). Sobra muita.
- **RAM é o limite.** Com 6 workers, 2 morreram ao carregar (SIGKILL da falta de RAM). Com 4, todos ficaram estáveis.
- **Vazão com 4 workers:** 8 frases de ~7,7 s (~62 s de áudio) em 41 s, ~1,5 s de áudio por segundo, ~11 s de GPU por parte. Na CPU local a mesma tarefa leva ~20 min.

O padrão é **`F5_WORKERS = "auto"`**: `min(RAM livre ÷ 2,5 GB, VRAM livre ÷ 2 GB, 8)`, o que dá ~4 na T4 grátis e mais em máquinas maiores (Colab Pro). Os workers carregam **um de cada vez**, porque o pico de RAM do carregamento era o que derrubava vários ao mesmo tempo. Para medir em vez de confiar no auto:

```bash
f5tts bench --voz ~/.claude/skills/f5-tts-voz/refs/theo --partes 8
# formato da saída (números ilustrativos):
# 8 partes em 41.0s · áudio 61.8s · 1.51s de áudio por segundo · pico de GPU 74%
```

Troque `WORKERS` na célula 1 por um número, rode de novo as células 1-3 e repita. Fique com o valor que dá mais *segundos de áudio por segundo*. O `f5tts status` mostra `ram.workers_rss_mb` (RAM de cada worker). Se o auto escolher mal, ajuste o `RAM_MB` do motor (`server/engines/<motor>.py`) ou o `VRAM_PER_WORKER_MB` do server com esses números.

### Quando um worker morre

- **Ao carregar:** não é recriado (quase sempre é falta de RAM e só repetiria). Ele aparece em `broken` com o motivo, por exemplo `exit -9 (SIGKILL: quase sempre falta de RAM)`, e o server segue com os outros. Reduza `F5_WORKERS`.
- **Depois de pronto:** o supervisor recria o worker, e a parte que ele gerava volta para a fila uma vez. Se morrer de novo na mesma parte, ela falha com o motivo. O `status` mostra os reinícios em `restarts`. Depois de 5 reinícios, o slot é dado como quebrado.

---

## Problemas comuns

| sintoma | causa / solução |
|---|---|
| célula 1: `assert gpu` | runtime sem GPU: *Runtime → Change runtime type → T4 GPU*. No plano grátis às vezes não há T4 disponível: tente mais tarde. |
| célula 1: erro no `import torchcodec` / `Could not load libtorchcodec` | o torchcodec instalado não casa com o torch do Colab. Instale a versão compatível com o `torch` impresso (tabela em <https://github.com/pytorch/torchcodec#installing-torchcodec>): `!pip install torchcodec==<versão>` e rode de novo da célula 1. |
| `broken: … SIGKILL` | falta de RAM: reduza `F5_WORKERS` (ou volte para `"auto"`). |
| `broken: <outro erro>` | o warmup falhou: veja a mensagem e o `/content/server.log` (torchcodec, checkpoint etc.). |
| `SERVER FORA DO AR ... HTTP 530/1033` | o túnel não tem conector: o Colab caiu ou a célula 3 não rodou. |
| `HTTP 401 token inválido` | o `F5_TOKEN` do Colab ≠ o do `f5tts config`. |
| `sem progresso há 900s` | worker travou: veja `f5tts status` e o log na célula 4. |
| `⚠ parte N ficou em X%` | o modelo errou a fala nas 3 tentativas. Normalmente é termo em inglês (ajuste o `--pronuncia`) ou frase difícil (reescreva). |
| sessão caiu no meio | o Colab grátis tem limite de ~12 h e derruba aba ociosa. O job em andamento se perde: rode Run all e gere de novo. |

---

## Desenvolvimento

```bash
# teste do client (sem modelo, ~2 s)
python3 tests/test_client.py

# server local na CPU, com o venv do f5-tts (lento, só para validar)
cd server && F5_TOKEN=dev F5_ENGINES=f5:2 F5_DEVICE=cpu F5_CKPT=~/.cache/f5tts-ptbr/model_last.safetensors \
  F5_JOBS_DIR=/tmp/f5jobs ~/.venvs/f5tts/bin/python run.py
F5_URL=http://127.0.0.1:8000 F5_TOKEN=dev f5tts status

# o mesmo, com o worker num venv diferente do server (como no Colab, com VENV=True no motor):
#   server num venv só com fastapi/uvicorn/numpy + ENGINE_PYTHON_f5=~/.venvs/f5tts/bin/python

# depois de mudar o server: commit + push, e no Colab rode de novo as células 1-3
git push
# só se mudar as células do notebook: regenere e suba o .ipynb de novo
python3 scripts/build_notebook.py
```

### API HTTP

Todas as rotas exigem o header `x-token`.

| rota | resposta |
|---|---|
| `GET /health` | `{default_engine, installed[], engines{nome: {workers, ready, loading, alive, wanted}}, engine_errors{}, workers, ready, loading, alive, broken{}, restarts{}, devices, pending_parts, asr{model, status, device, rss_mb}, ram_budget_mb, idle_unload_secs, ram, gpu}` |
| `POST /jobs` `{texts[], ref_audio (wav base64), ref_text, engine?, nfe=32, speed=1.0, opts{}, transcribe=false}` | `{job_id, engine, engine_ready, total}`; um `engine` instalado mas não carregado é carregado na hora |
| `GET /jobs/{id}` | `{engine, engine_ready, status: queued\|running\|done\|error, total, done[], failed{}, gen_secs, audio_secs, heard{n: texto}}` (`heard` só com `transcribe`) |
| `GET /jobs/{id}/parts/{n}` | wav mono da parte `n` (quando pronta), na taxa do motor |
| `DELETE /jobs/{id}` | apaga o job (partes ainda na fila são descartadas) |

Estilos, dicionário e conferência ficam no client: um roteiro com estilos vira um job por estilo, todos na mesma fila.

Configuração do server por variáveis de ambiente:

| variável | |
|---|---|
| `F5_TOKEN` | obrigatório |
| `F5_INSTALLED` | motores que podem ser pedidos: `f5,qwen3,chatterbox` (padrão `f5`) |
| `F5_DEFAULT_ENGINE` | motor carregado ao subir e usado sem `engine` (padrão: o 1º instalado) |
| `F5_WORKERS` | workers do motor padrão: `auto` (cabe na RAM/VRAM) ou um número |
| `F5_IDLE_UNLOAD` | segundos que um motor sob demanda fica carregado sem uso (padrão 600) |
| `F5_ASR_MODEL` | modelo do faster-whisper do worker de ASR (padrão `small`); `off` desliga |
| `F5_ENGINES` | legado: `f5:auto,qwen3:1` vira instalados = os nomes, padrão = o 1º, workers = o número dele |
| `ENGINE_PYTHON_<motor>` | Python dos workers daquele motor (venv próprio); padrão: o mesmo do server |
| `F5_DEVICE` | `cuda`/`cpu` (padrão: o motor escolhe) |
| `F5_JOBS_DIR`, `PORT=8000` | |
| `F5_MODEL`, `F5_CKPT` | do motor F5 (outros motores têm as suas, veja `engines/<motor>.py`) |

**Novo motor:** crie `server/engines/<nome>.py` seguindo o contrato de `engines/__init__.py` (`NAME`, `SAMPLE_RATE`, `PIP`, `VENV`, `load`, `warmup`, `synth`), acrescente o nome em `AVAILABLE`, faça push e use o nome em `ENGINES` no notebook. Um motor com dependências que brigam com as dos outros usa `VENV = True`, e o notebook instala ele num venv próprio.
