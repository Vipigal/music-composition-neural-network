---
type: Reference
date: 2026-11-23
question: Por que o gerador soa sem estrutura (jazz como solo, funk que deriva) e o que a literatura de geração de bateria e música simbólica faz e mede sobre estrutura de frase, repetição e deriva?
sources:
  - id: groovae
    resource: https://arxiv.org/abs/1905.06118
    title: Gillick et al., Learning to Groove with Inverse Sequence Transformations (ICML 2019)
  - id: gmd
    resource: https://magenta.tensorflow.org/datasets/groove
    title: Magenta, Groove MIDI Dataset (2019)
  - id: jazz-transformer
    resource: https://arxiv.org/abs/2008.01307
    title: Wu & Yang, The Jazz Transformer on the Front Line - Exploring the Shortcomings of AI-composed Music through Quantitative Measures (ISMIR 2020)
  - id: holtzman
    resource: https://arxiv.org/abs/1904.09751
    title: Holtzman et al., The Curious Case of Neural Text Degeneration (ICLR 2020)
  - id: music-transformer
    resource: https://arxiv.org/abs/1809.04281
    title: Huang et al., Music Transformer (2018)
  - id: remi
    resource: https://arxiv.org/abs/2002.00212
    title: Huang & Yang, Pop Music Transformer (ACM MM 2020)
  - id: makris2022
    resource: https://arxiv.org/abs/2202.04464
    title: Makris et al., Conditional Drums Generation using Compound Word Representations (2022)
  - id: dahale2022
    resource: https://arxiv.org/abs/2209.00291
    title: Dahale et al., Generating Coherent Drum Accompaniment with Fills and Improvisations (ISMIR 2022)
  - id: nuttall2021
    resource: https://nime.org/proc/nime2021_33/
    title: Nuttall, Haki & Jordà, Transformer Neural Networks for Automated Rhythm Generation (NIME 2021)
---

# Estrutura e deriva na bateria gerada

Motivação: a revisão de um baterista em 2026-11-23 disse que o jazz gerado soa como "seção de solo
de bateria, e não uma parte de condução", e que o funk "começa bem e depois despiroca". O modelo é o
Transformer decoder-only de 3,4 M do spec (`docs/specs/gerador-bateria.md`), com amostragem em
temperatura 1,0, sem top-p, e CFG 3.

## TL;DR

- A literatura conhece o problema. O Jazz Transformer resume as falhas de um Transformer em jazz
  como "erraticidade" e "ausência de estruturas repetitivas de médio e longo prazo". O GrooVAE
  nem chega a enfrentá-lo: avalia só trechos de 2 compassos.
- Nos nossos dados, a crítica se divide em duas:
  - **Funk e rock:** é deriva de amostragem. Com temperatura 0,9 e top-p 0,9, a repetição entre
    compassos chega ao nível humano.
  - **Jazz:** a variação entre compassos já é igual à humana, porque o jazz do GMD é improvisado.
    O que falta é a **condução**: o ride aparece em 25% dos tempos, contra 43% no humano, e baixar
    a temperatura piora isso.
- No prazo cabem três coisas: a varredura de amostragem, as métricas de estrutura e condução no
  T10, e a geração com prompt. Atenção relativa e noção de frase ficam como trabalho futuro.

## Achados

1. O GrooVAE avalia só padrões de 2 compassos, "a typical length for drum loops", embora tenha
   explorado até 16. Ele não diz nada sobre estrutura de longo prazo. A sua hipótese ("o GrooVAE
   deve ter mencionado isso") é **refutada no sentido estrito**: ele contorna o problema, não o
   discute.
2. O GMD é improvisado por construção. Os bateristas foram instruídos a tocar sequências longas
   ("several minutes of continuous playing") e "were able to improvise in a wide range of styles".
   O gênero é rotulado pelo próprio baterista.
3. O Jazz Transformer (Transformer-XL, jazz) mede a falta de estrutura repetitiva: no modelo, "all
   traces of repetitive structures disappear at the timescale of 10 seconds". Ele propõe o
   *structureness indicator* (fitness scape plot sobre uma matriz de auto-similaridade) e a
   *groove similarity* entre compassos.
4. A amostragem pura (temperatura 1, sem truncar) gera texto "incoherent and almost unrelated to
   the context" por causa da "unreliable tail" (Holtzman). Top-p e temperatura abaixo de 1 trocam
   diversidade por coerência.
5. Continuar a partir de um prompt real (*priming*) é o protocolo de avaliação do Music
   Transformer. Ele também mostra que a posição absoluta generaliza mal para além do comprimento
   de treino, e a atenção relativa resolve isso (exige retreino).
6. Em bateria, a consistência de padrão entre compassos é tratada como requisito (Dahale et al.
   2022, Makris et al. 2022, com a *Groove Consistency* entre compassos vizinhos).

## Por eixo

### Eixo 1 — GrooVAE e GMD

- **Comprimento.** "though we explored models for sequences of up to 16 measures, for consistency
  we use 2 measure (or 2 bar) patterns for all reported experimental evaluations" e "We chose 2
  measures [...] because 2 bars is a typical length for drum loops used in music production
  practice" (GrooVAE §3.2, conferido em 2026-11-23).
- **"Groove" no GrooVAE é só microtiming e velocity** ("performance characteristics or groove",
  §3.1). Condução e padrão de estilo não entram, e não há discussão de solo nem de condução.
  Fonte: o relatório do agente, sem conferência literal.
- **Avaliação.** Teste de escuta "sounds more like a human drummer" e métricas de distribuição de
  timing. Nenhuma métrica de estrutura (relatório do agente).
- **Coleta do GMD.**
  - "The professionals were able to improvise in a wide range of styles, resulting in a diverse
    dataset."
  - "The drummers were instructed to play a mix of long sequences (several minutes of continuous
    playing) and short beats and fills."
  - "Each performance is annotated with a genre (provided by the drummer)".
  - Página do GMD, conferida em 2026-11-23.
- **Trabalhos posteriores no GMD.**
  - O resumo do Nuttall, Haki & Jordà (NIME 2021) afirma geração "comparable to human playing
    across arbitrarily long time periods" com Transformer-XL. O corpo não foi lido: é afirmação
    dos autores, não medição verificada.
  - Não encontramos trabalho que meça deriva em Transformer treinado no GMD.

### Eixo 2 — Estrutura e deriva em geração simbólica

- **Diagnóstico.** "we can summarize the model's deficiencies as follows: 1) the erraticity of the
  generated musical events; and, 2) the absence of medium- and long-term repetitive structures"
  (Jazz Transformer, §6, conferido).
- **Métricas.**
  - *Groove similarity* (Jazz Transformer, §5.2): 1 − média do XOR dos vetores binários de onset
    por compasso.
  - *Structureness indicator* (§5.4): máximo do fitness scape plot numa faixa de durações, a
    partir da matriz de auto-similaridade. O cálculo da fitness vem de uma referência que não lemos.
  - *Compression ratio* (Makris 2022): mais compressível significa mais padrão.
  - Ressalva de Makris: métricas de baixo nível "cannot guarantee to accurately measure the
    quality".
- **Amostragem.** Holtzman §1 e §3, conferido: a amostragem pura falha pela cauda. O nucleus
  sampling corta a cauda, e a temperatura baixa "comes at the cost of decreasing diversity". O
  mecanismo foi medido em texto. Em bateria, medimos nós (ver abaixo).
- **Prompt e posição.**
  - O Music Transformer avalia continuação de *priming sequences*.
  - Com atenção relativa, o modelo "elaborates the motif and creates phrases with clear contour".
  - A posição absoluta, que é a nossa, generaliza mal além do comprimento de treino.
  - Fonte: relatório do agente.
- **Tokens métricos.** `Bar` e `Position` dão "a metrical context for models to count the beats".
  Sem eles, as barras de compasso derivam (REMI). O PerTok já tem os dois, então a nossa deriva é
  de conteúdo, não de alinhamento.

## Medição no nosso caso (2026-11-23)

Script ad hoc no scratchpad, sobre o `best.pt` de `runs/20261123-1410_transformer`.
- **Amostras:** 50 por configuração. Os humanos são as janelas de referência de validação de 8
  compassos (jazz 11, funk 43, rock 76).
- **GS vizinhos:** groove similarity média entre compassos consecutivos, numa grade de 16
  semicolcheias, com qualquer peça.
- **Condução:** fração dos tempos com ride, chimbal fechado ou chimbal aberto.

| Estilo | Sistema | GS vizinhos | Condução |
|---|---|---|---|
| jazz | humano | 0,720 | 0,84 |
| jazz | Markov | 0,588 | 0,58 |
| jazz | Transformer, T 1,0, CFG 3 | 0,715 | 0,54 |
| jazz | Transformer, T 0,9, top-p 0,9, CFG 3 | 0,816 | 0,46 |
| funk | humano | 0,898 | 0,97 |
| funk | Markov | 0,531 | 0,73 |
| funk | Transformer, T 1,0, CFG 3 | 0,833 | 0,96 |
| funk | Transformer, T 0,9, top-p 0,9, CFG 3 | 0,911 | 0,99 |
| rock | humano | 0,831 | 0,76 |
| rock | Markov | 0,556 | 0,69 |
| rock | Transformer, T 1,0, CFG 3 | 0,842 | 0,76 |
| rock | Transformer, T 0,9, top-p 0,9, CFG 3 | 0,928 | 0,77 |

No jazz, o CFG não explica a falta de condução:

| Jazz | GS vizinhos | Condução | Ride |
|---|---|---|---|
| humano | 0,720 | 0,84 | 0,43 |
| sem CFG, T 1,0 | 0,763 | 0,62 | 0,25 |
| CFG 1,5, T 1,0 | 0,748 | 0,56 | 0,24 |
| CFG 3, T 1,0 | 0,715 | 0,54 | 0,28 |
| CFG 3, T 0,9, top-p 0,9 | 0,816 | 0,46 | 0,26 |

Leitura:
- O Markov fica longe do humano na repetição (0,53 a 0,59 em todos os estilos). O Transformer
  fica perto.
- **Funk:** a deriva que o baterista ouviu aparece como GS 0,833 contra 0,898 do humano. T 0,9 com
  top-p 0,9 leva a 0,911, praticamente o humano.
- **Jazz:** a "falta de estrutura" não está na repetição, que é igual à humana (o jazz humano do
  GMD é o estilo que mais varia). Está na **condução**: o modelo toca ride e chimbal em 54 a 62%
  dos tempos, contra 84% do humano. Baixar a temperatura aumenta a repetição e reduz a condução,
  ou seja, o modelo repete outra coisa que não o ride.
- **Rock:** com T 1,0 já está no nível humano. T 0,9 deixa mais repetitivo que o humano.

## Aplicação ao projeto

Em ordem de custo-benefício, com 1 dia até a entrega:

1. **No T10, varrer a amostragem na validação.**
   - Grade: temperatura {0,9; 1,0} × top-p {0,9; nenhum} × CFG {1; 1,5; 3}.
   - Escolher por estilo ou um valor global, olhando GS e condução contra o humano, além das
     métricas D7.
   - Pelos números acima, funk e rock ganham com T 0,9 e top-p 0,9, e o jazz não.
   - Custo: minutos de GPU. Exige incluir temperatura e top-p na varredura, que hoje só cobre o CFG.
2. **Acrescentar GS vizinhos, GS com o 1º compasso e condução às métricas D7.** Isso põe a crítica
   do baterista em número no artigo, com o Markov como contraste. Custo: cerca de 45 min, sem GPU.
3. **Gerar a partir de um prompt** de 1 a 2 compassos humanos de validação. É o protocolo do Music
   Transformer e a tarefa (a) do enunciado. É o mais promissor para o jazz: um prompt com ride
   estabelece a condução que o modelo, sozinho, sub-representa. Custo: cerca de 1 h.
4. **Trabalho futuro:**
   - atenção relativa (Music Transformer);
   - embedding de posição na frase;
   - mais dados de condução de jazz, ou separar condução e solo nos rótulos;
   - avaliação com structureness indicator.

## Anti-padrões

- **Escolher a dedo depois de ouvir para a avaliação.** Vale para a demo, se rotulado como
  selecionado. Não vale para as métricas nem para as 5 amostras do entregável (spec §6.7).
- **Baixar a temperatura como solução universal.** No jazz, isso aumenta a repetição sem trazer a
  condução de volta.
- **Medir estrutura em valor absoluto.** Sempre compare com o humano do mesmo estilo: o jazz humano
  tem GS 0,72, e o funk, 0,90.

## Questões em aberto

- Por que o modelo sub-representa o ride no jazz?
  - Hipóteses: os 39 grooves de jazz no treino; a ponderação 1/√ que mistura estilos; o chimbal de
    pé (44, mapeado para 42) contado como condução no humano.
  - O que resolveria: medir o ride no treino de jazz por performance, e a perda por classe de token.
- O texto completo do Nuttall et al. 2021 (NIME) não foi lido. Ele afirma geração coerente longa
  com Transformer-XL no GMD e seria o melhor contraponto no artigo.

## Fontes

Acessadas em 2026-11-23.

- Gillick et al., Learning to Groove with Inverse Sequence Transformations, ICML 2019 — https://arxiv.org/abs/1905.06118
- Magenta, The Groove MIDI Dataset — https://magenta.tensorflow.org/datasets/groove
- Wu & Yang, The Jazz Transformer on the Front Line, ISMIR 2020 — https://arxiv.org/abs/2008.01307
- Holtzman et al., The Curious Case of Neural Text Degeneration, ICLR 2020 — https://arxiv.org/abs/1904.09751
- Huang et al., Music Transformer, 2018 — https://arxiv.org/abs/1809.04281
- Huang & Yang, Pop Music Transformer, ACM MM 2020 — https://arxiv.org/abs/2002.00212
- Makris et al., Conditional Drums Generation using Compound Word Representations, 2022 — https://arxiv.org/abs/2202.04464
- Dahale et al., Generating Coherent Drum Accompaniment with Fills and Improvisations, ISMIR 2022 — https://arxiv.org/abs/2209.00291
- Nuttall, Haki & Jordà, Transformer Neural Networks for Automated Rhythm Generation, NIME 2021 — https://nime.org/proc/nime2021_33/ (só o resumo)
