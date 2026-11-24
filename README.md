# drumgen: gerador de bateria condicionado por estilo

Gera padrões de bateria expressivos (com microtiming e dinâmica) condicionados por estilo musical, treinados no
[Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) (GMD). O sistema principal é um Transformer
autoregressivo com *classifier-free guidance* (CFG) por estilo; o baseline é uma cadeia de Markov por estilo.

Estilos avaliados: `rock`, `funk`, `latin`, `jazz`, `hiphop` e `soul` (o resto do GMD vira `other`).

O pipeline tem cinco etapas, todas rodando como módulo Python com uma config YAML:

```
prepare → train / markov → generate → render → evaluate
```

## 1. Ambiente

Precisa de Python 3.12 e, para gerar áudio, do `fluidsynth` com um soundfont General MIDI.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# áudio (Ubuntu/Debian/WSL)
sudo apt install fluidsynth fluid-soundfont-gm
```

O soundfont padrão é `/usr/share/sounds/sf2/FluidR3_GM.sf2` (vem com o `fluid-soundfont-gm`). Para usar outro, mude
`render.soundfont` na config.

O treino usa `cuda` por padrão. Sem GPU, coloque na config:

```yaml
training:
  device: cpu
```

## 2. Dados

Baixe a versão só-MIDI do GMD e descompacte em `data/raw/groove`:

```bash
mkdir -p data/raw && cd data/raw
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
unzip groove-v1.0.0-midionly.zip   # cria data/raw/groove/ com info.csv e as pastas drummer*
cd ../..
```

`data/` e `runs/` não vão para o git.

## 3. Rodando o pipeline

### 3.1 Preparar (tokenizar)

```bash
python -m drumgen.prepare --config configs/transformer.yaml
```

Grava em `data/processed/`:

| Arquivo | O que é |
| --- | --- |
| `tokenizer.json` | Vocabulário do tokenizer (usado por todas as etapas seguintes) |
| `train.npz`, `val.npz`, `test.npz` | Sequências de tokens de cada split |
| `pieces.csv` | Uma linha por peça do GMD: estilo, tipo (groove/virada), BPM, compassos, split |
| `stats.json` | Estatísticas do corpus, incluindo o BPM mediano de cada estilo (usado no render) |
| `reference/<split>/<estilo>/*.mid` | Janelas de 8 compassos de bateria humana, usadas como referência na avaliação |

### 3.2 Treinar

Transformer:

```bash
python -m drumgen.train --config configs/transformer.yaml
```

Baseline Markov (escolhe a ordem pela NLL de validação):

```bash
python -m drumgen.markov --config configs/markov.yaml
```

Cada treino cria uma pasta de run `runs/<AAAAMMDD-HHMM>_<nome da config>/`:

| Arquivo | O que é |
| --- | --- |
| `config.yaml` | Config completa usada + commit do git (para reproduzir) |
| `best.pt` | Checkpoint do Transformer na época de menor loss de validação (é o que a geração usa) |
| `last.pt` | Checkpoint da última época treinada |
| `history.csv` | Uma linha por época: loss de treino, de validação (geral e por estilo), LR e segundos |
| `curves.png` | Curvas de treino e validação, com a melhor época marcada |
| `markov.npz`, `markov_nll.csv` | (só no Markov) a cadeia treinada e a NLL de cada ordem candidata |

O treino para sozinho por *early stopping* (`training.patience` épocas sem melhorar a validação).

**Teste rápido:** `configs/debug.yaml` usa 12 peças por split e 3 épocas, para verificar que tudo roda em poucos
minutos. Rode o `prepare` e o `train` com ela (os dados vão para `data/processed_debug`).

### 3.3 Gerar

```bash
python -m drumgen.generate --config configs/transformer.yaml \
    --system transformer --model runs/<run_do_transformer>
```

- `--system` é `transformer` ou `markov`.
- `--model` é a pasta da run treinada (onde estão o `best.pt` ou o `markov.npz`).
- `--run` (opcional) é a run onde as amostras serão gravadas. Por padrão é a própria `--model`. Para comparar os dois
  sistemas na mesma avaliação, gere o Markov dentro da run do Transformer:

```bash
python -m drumgen.generate --config configs/markov.yaml \
    --system markov --model runs/<run_do_markov> --run runs/<run_do_transformer>
```

A saída fica em `runs/<run>/samples/<sistema>/`:

```
samples/transformer/
├── manifest.csv
├── rock/0.mid, 1.mid, ...           # amostra original (o nome é a seed)
├── rock_quantizado/0.mid, ...       # a mesma amostra com o microtiming removido (grade de semicolcheia)
├── funk/ ...                        # uma pasta por estilo avaliado
├── troca_rock-funk/ ...             # 4 compassos de rock seguidos de 4 de funk
└── rock_cfg1.5/ ...                 # só com cfg_scale_sweep: uma pasta por escala de CFG
```

O `manifest.csv` tem uma linha por amostra com sistema, pasta, estilo, seed, parâmetros de amostragem (CFG,
temperatura, top-k, top-p), número de tokens, se terminou antes da hora (`ended_early`), notas descartadas na
decodificação e o caminho do `.mid`. Amostras que terminam antes dos 8 compassos ficam registradas sem arquivo e
são refeitas com a próxima seed.

Se a pasta `samples/<sistema>/` já tiver amostras, a geração recusa; apague-a ou grave em outra run.

**Ajustando a geração:** os parâmetros ficam no bloco `generation` da config. Por exemplo, crie um
`configs/minha_geracao.yaml`:

```yaml
name: minha_geracao
generation:
  samples_per_style: 10     # padrão 100
  cfg_scale: 3.0            # 1.0 = sem guidance; maior = estilo mais marcado
  temperature: 0.9
  top_p: 0.95
  style_switches:
    - {before: jazz, after: hiphop}
  # cfg_scale_sweep: [1.0, 1.5, 3.0]   # se preenchido, gera SÓ a varredura de CFG
```

O Markov não usa CFG, temperatura nem top-k/top-p, e não gera troca de estilo. A variante de virada (`fill_styles`)
ainda não é suportada por nenhum dos dois sistemas. Variantes não suportadas são puladas com um aviso.

### 3.4 Renderizar (MIDI → WAV + piano-roll)

```bash
python -m drumgen.render --config configs/transformer.yaml runs/<run>/samples/transformer/rock
```

Aceita arquivos `.mid` ou pastas (procura `.mid` recursivamente). Ao lado de cada `.mid` grava:

- `<seed>.wav`: áudio mono, tocado no BPM mediano do estilo (tirado de `stats.json`);
- `<seed>.png`: piano-roll da bateria, com o título e o BPM.

Para comparar com bateria humana, renderize também as referências:

```bash
python -m drumgen.render --config configs/transformer.yaml data/processed/reference/val/rock
```

### 3.5 Avaliar

```bash
python -m drumgen.evaluate --config configs/eval.yaml --run runs/<run>
```

Compara todas as pastas em `runs/<run>/samples/` com as janelas humanas de `data/processed/reference/val/`. Para usar
o conjunto de teste, faça uma config com `evaluation: {reference_split: test}`.

A saída fica em `runs/<run>/eval/`:

| Arquivo | O que é |
| --- | --- |
| `summary.csv` | Por sistema, variante, estilo e métrica: média e desvio das amostras e da referência, distância entre médias, *overlap area* e divergência KL |
| `per_sample.csv` | O valor de cada métrica para cada `.mid` (geradas e referência) |
| `style_matrix.csv` | Matriz de estilo: o quanto as amostras de cada estilo se parecem com a referência de cada estilo |
| `distribution_<métrica>.png` | Distribuições de cada métrica, gerado × humano |
| `style_matrix_<sistema>_<variante>.png` | A matriz de estilo em forma de heatmap |

Métricas: `empty_beat_rate`, `groove_consistency`, `drum_pattern_consistency`, `density`, `swing` e
`microtiming_deviation`.

## 4. Ouvindo as músicas

O jeito mais simples é renderizar (seção 3.4) e abrir o `.wav` em qualquer player.

**No WSL**, para abrir no player do Windows:

```bash
explorer.exe "$(wslpath -w runs/<run>/samples/transformer/rock/0.wav)"
```

ou abra a pasta inteira no Explorer e clique nos arquivos:

```bash
explorer.exe "$(wslpath -w runs/<run>/samples/transformer/rock)"
```

**No Linux com áudio:**

```bash
xdg-open runs/<run>/samples/transformer/rock/0.wav
# ou, no terminal:
ffplay -nodisp -autoexit runs/<run>/samples/transformer/rock/0.wav
```

**Direto do MIDI, sem renderizar:**

```bash
fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 runs/<run>/samples/transformer/rock/0.mid
```

No WSL, adicione `-a pulseaudio` se o fluidsynth não achar a saída de áudio. Os `.mid` também abrem em qualquer DAW
ou editor de partitura (MuseScore, Reaper, GarageBand). A faixa é marcada como bateria (canal 10), então toca com
sons de percussão.

Para ouvir o efeito do microtiming, compare `rock/0.wav` com `rock_quantizado/0.wav`: é a mesma amostra, só que
presa na grade.

## 5. Testes

```bash
pytest
pyright
```

## Estrutura

```
configs/              configs YAML (transformer, markov, eval, debug)
src/drumgen/
├── config.py         dataclasses de configuração, estilos, runs
├── tokenizer.py      tokenizer de bateria (MidiTok) com microtiming e velocity
├── prepare.py        GMD → tokens + janelas de referência
├── dataset.py        janelas de treino com dropout de estilo para o CFG
├── model.py          o Transformer (com cache de K/V para geração)
├── sampling.py       amostragem: temperatura, top-k, top-p, CFG, plano por compasso
├── train.py          laço de treino, early stopping, checkpoints e curvas
├── markov.py         baseline Markov
├── generate.py       geração em lote + manifest
├── render.py         MIDI → WAV (fluidsynth) e piano-roll
├── metrics.py        métricas de groove
└── evaluate.py       comparação com a referência humana
notebooks/            análise exploratória do dataset (figuras em results/dataset/)
research/             notas de pesquisa
tests/                testes (pytest)
```
