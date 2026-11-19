---
type: Reference
date: 2026-11-19
question: Como representar swing e microtiming, extrair o máximo do Groove MIDI Dataset e condicionar a geração por estilo?
sources:
  - id: groovae
    resource: https://arxiv.org/abs/1905.06118
    title: Gillick et al., Learning to Groove with Inverse Sequence Transformations (ICML 2019)
  - id: haki2022
    resource: https://zenodo.org/records/7088343
    title: Haki et al., Real-Time Drum Accompaniment Using Transformer Architecture (AIMC 2022)
  - id: pertok
    resource: https://arxiv.org/abs/2410.02060
    title: Lenz & Mani, PerTok - Expressive Encoding and Modeling of Symbolic Musical Ideas and Variations (2024)
  - id: musicgen
    resource: https://arxiv.org/html/2306.05284
    title: Copet et al., Simple and Controllable Music Generation (2023)
  - id: cfg-lm
    resource: https://arxiv.org/abs/2306.17806
    title: Sanchez et al., Stay on topic with Classifier-Free Guidance (2023)
  - id: egmd
    resource: https://magenta.tensorflow.org/datasets/e-gmd
    title: Magenta, The Expanded Groove MIDI Dataset (2020)
  - id: gmd
    resource: https://magenta.tensorflow.org/datasets/groove
    title: Magenta, Groove MIDI Dataset (2019)
  - id: makris2022
    resource: https://arxiv.org/abs/2202.04464
    title: Makris et al., Conditional Drums Generation using Compound Word Representations (2022)
  - id: lm-drummers
    resource: https://arxiv.org/abs/2301.01162
    title: Zhang & Callison-Burch, Language Models are Drummers (AAAI CreativeAI Workshop 2023)
  - id: nuttall2021
    resource: https://nime.org/proc/nime2021_33/
    title: Nuttall, Haki & Jordà, Transformer Neural Networks for Automated Rhythm Generation (NIME 2021)
  - id: muspy
    resource: https://muspy.readthedocs.io/en/latest/metrics.html
    title: MusPy, documentação de métricas
---

# Microtiming, dados e condicionamento para geração de bateria

Pesquisa feita em 2026-11-19 para fundamentar o desenho do gerador de bateria condicionado por
estilo, treinado no Groove MIDI Dataset (GMD). Une o que as fontes dizem e o que medimos nos dados.

## TL;DR

- A representação passa a ser **grade de semicolcheia mais um token de microtiming por nota**
  (PerTok, já disponível no `miditok`). A grade de 1/12 de tempo apaga o swing intermediário do
  jazz. Separar "posição na grade" de "desvio" é o que a literatura de bateria faz (GrooVAE).
- A avaliação passa a medir **timing e velocity por posição métrica** (KL, como no GrooVAE) e a
  **razão de swing por estilo**. As métricas do muspy não enxergam microtiming.
- **CFG** segue a receita publicada do MusicGen: condição descartada com probabilidade 0,2 no
  treino e escala de guidance 3,0 na amostragem.
- **Não há dado simbólico extra**: o E-GMD regrava as mesmas sequências do GMD. Os fills
  (921 compassos) são o único ganho de dados dentro do próprio dataset.

## Achados com fonte

### Representação de microtiming

- GrooVAE: "we take 16th notes as the fundamental timestep of our data", com offsets "taking
  values in [-0.5, 0.5) that indicate how far and in which direction each note's timing lies
  relative to the nearest 16th note" [groovae, §3.2–3.3]. Usa 9 categorias de bateria e matrizes
  de hits, velocities e offsets.
- Haki et al. usam a mesma representação do GrooVAE num Transformer em tempo real [haki2022].
  **Não confirmado em leitura**: só resumo de busca.
- PerTok codifica cada nota com TimeShift quantizado mais um token MicroTiming com o resto:
  "Microshift 15 represents a placement of 15 ticks after the quantized note position"
  [pertok, §2.2]. O artigo não fixa o número de bins. A implementação está em `miditok.PerTok`
  (miditok 3.1.0), com `use_microtiming`, `max_microtiming_shift` e `num_microtiming_bins`.
- **Lacuna:** nenhuma fonte compara de forma medida "grade fina" contra "grade + offset".

### Avaliação de timing

- GrooVAE agrupa as notas pelas 4 posições de semicolcheia e calcula o KL entre as distribuições
  de timing e de velocity real e gerada: "we aggregate all predicted notes into four groups based
  on which 16th note position they align with" [groovae, §7]. Reporta também o Timing MAE: 18,47 ms
  (Seq2Seq + VIB) contra 22,34 ms (KNN).
- PerTok mede o microtiming "as a percentage deviation from the nearest 16th note", em 100 bins
  [pertok, §4.2].
- Nenhuma das fontes lidas define uma métrica de swing. A razão de swing por estilo é proposta nossa.
- muspy [muspy]: `drum_in_pattern_rate` aceita só `meter` duple ou triple. `drum_pattern_consistency`
  é o maior dos dois. `groove_consistency` exige compasso constante e devolve NaN com menos de
  2 compassos. As métricas de pitch (`polyphony`, `pitch_*`, `scale_*`) ignoram faixas de bateria.

### Condicionamento e CFG

- MusicGen: "during training we drop the condition with a probability of 0.2" e "during inference
  we use a guidance scale of 3.0" [musicgen, §3.1].
- CFG em LM autoregressivo puro "brings improvements equivalent to a model with twice the
  parameter-count" [cfg-lm, abstract].
- **Lacuna:** não achamos comparação medida entre token de controle prefixado e embedding somado
  por posição.

### Dados

- E-GMD: "re-recording the GMD sequences on 43 drumkits using a Roland TD-17", "We maintained the
  same train, test and validation splits", e "If your application requires only symbolic drum
  data, we recommend using the original data from the Groove MIDI Dataset" [egmd].
- GMD: 13,6 h, 1.150 arquivos, mais de 22.000 compassos. A página não descreve o critério da
  partição. Diz apenas que as `eval_session` de 4 bateristas estão no teste [gmd].
- **Lacuna:** não achamos medição de ganho por data augmentation em bateria simbólica.

## Medições locais (GMD, scripts rodados em 2026-11-19)

Escopo: beats em 4/4. São 491 performances, 12,7 h e 22.043 compassos, dos quais 17.670 são de treino.

### Swing e microtiming por estilo

"Contratempo" é a posição mediana, dentro do tempo, das notas entre 0,4 e 0,8 do tempo. O valor
0,5 é colcheia reta e 0,667 é tercina.

| Gênero | Contratempo | Microtiming mediano (vs. 16avo) | Notas em posição de tercina |
|---|---|---|---|
| jazz | 0,583 | 21,9 ms | 12,2% |
| rock | 0,535 | 21,2 ms | 9,4% |
| latin | 0,510 | 21,7 ms | 11,6% |
| funk | 0,508 | 18,5 ms | 8,0% |
| hiphop | 0,500 | 19,6 ms | 6,2% |
| soul | 0,488 | 19,2 ms | 4,8% |

O swing do jazz (cerca de 1,4:1) fica entre os pontos 0,500 e 0,667 da grade de 1/12, por isso
essa grade o apaga.

### Tokenização (120 arquivos, 9 classes, 16 níveis de velocity)

| Representação | Tokens/compasso | Erro mediano de onset | Vocabulário |
|---|---|---|---|
| REMI, grade 1/4 de tempo | 51 | 21,8 ms | 157 |
| REMI, grade 1/12 | 54,0 | 11,8 ms | 197 |
| REMI, grade 1/24 | 55,4 | 6,0 ms | 245 |
| REMI, grade 1/48 | 56,7 | 2,9 ms | 341 |
| PerTok, grade 1/16 + 16 bins de microtiming (±1/8 de tempo) | 69,7 | 2,3 ms | 181 |
| PerTok, grade 1/16 + 32 bins de microtiming | 69,7 | 1,1 ms | 197 |

Uma matriz no estilo do GrooVAE, com um hit por classe por semicolcheia, perderia 8,7% das notas.

### Fills

- 647 arquivos, 921 compassos (+5,2% sobre os compassos de beat de treino). A mediana é de
  1,04 compasso por arquivo. 70% começam no tempo 1 e só 8% trazem o "pouso" no compasso seguinte.
- Têm densidade menor que os beats (13,3 contra 17,5 notas por compasso) e muito mais toms
  (29% das notas contra 8%).
- 100% dos fills têm um beat da mesma sessão, estilo e BPM, mas só 72,5% têm esse beat na mesma
  partição.
- 10,3% dos compassos de beat de treino já têm 25% ou mais de toms, ou seja, viradas sem rótulo
  dentro das performances. A heurística "fração de toms" recupera só 54% dos compassos de fill
  rotulados com limiar de 0,25, e 72% com limiar de 0,15.

### muspy em bateria

Num arquivo de funk, `drum_in_pattern_rate` sem quantizar deu 0,03, e quantizado em 1/12 deu
0,58. `polyphony` deu NaN e `n_pitches_used` deu 0.

## Não confirmado

Os papers abaixo foram vistos só em resultados de busca. Ler antes de citar:

- Agogic / PMT (arXiv 2608.03999, 2026);
- BEAT (arXiv 2604.19532, 2026);
- GrooveTransformer em Eurorack (NIME 2024);
- corpo de Nuttall et al. 2021 [nuttall2021] e de Makris et al. 2022 [makris2022]. Do Makris lemos
  só que é condicionado, com Compound Word, sobre um subconjunto rock/metal do Lakh, sem velocity.
