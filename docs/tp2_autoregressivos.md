Universidade Federal de Minas Gerais
Instituto de Ciências Exatas
Departamento de Ciência da Computação
DCC831 – IA Generativa para Música (2026/2)

# Projeto Final – Geração Musical Simbólica com Redes Neurais Autoregressivas

**Data de entrega:** [25/11/2026], 23:59h

**Trabalho em grupo:** 3-5 integrantes

## 1 Introdução

O objetivo deste trabalho é dar experiência prática com modelos generativos baseados em **redes neurais autoregressivas** para música simbólica. Cada grupo deve definir uma tarefa de geração musical, escolher um dataset MIDI, representar as músicas como sequências de *tokens*, treinar um modelo neural autoregressivo e compará-lo com um *baseline* mais simples, avaliando as músicas geradas de forma objetiva em relação às músicas humanas do dataset. Assim como no TP1, o trabalho também serve de treino para a escrita científica: o projeto deverá ser reportado em um artigo curto utilizando o formato de artigo do ISMIR.

## 2 Instruções

1. **Tarefa de geração.** Escolha uma tarefa de geração musical simbólica e explicite-a no artigo. Exemplos:
   - (a) Composição livre (incondicional), a partir do zero ou continuando um trecho inicial (*prompt*);
   - (b) Composição condicionada por atributos (gênero, emoção, compositor, tonalidade etc.);
   - (c) *Text2music*: geração de MIDI a partir de uma descrição textual;
   - (d) Arranjo automático: dada uma melodia, gerar acompanhamento ou as demais vozes/instrumentos;
   - (e) Outras tarefas (harmonização, bateria, *infilling* etc.), mediante aprovação do professor.

2. **Dataset.** Escolha um dataset MIDI adequado à tarefa. Sugestões:
   - Nottingham e JSB Chorales (melodias e harmonização)
   - MAESTRO
   - GiantMIDI-Piano (piano solo)
   - POP909 (arranjo de piano pop)
   - EMOPIA (condicionamento por emoção)
   - Lakh MIDI Dataset (multi-instrumento, gênero)
   - Groove MIDI Dataset (bateria)
   - MidiCaps (*text2music*)

   Os dados devem ser separados em conjuntos de **treino**, **validação** e **teste** (use a partição oficial quando existir). A separação deve ser feita **por música**, e não por trecho, para evitar vazamento de dados entre os conjuntos. O conjunto de teste deve ser usado apenas na avaliação final. O pré-processamento (filtragem, quantização, transposição/*data augmentation*, recorte em janelas etc.) e as estatísticas do dataset devem ser descritos no artigo.

3. **Representação.** Escolha uma forma de codificar o MIDI como uma sequência de *tokens* (ex.: REMI, MIDI-Like, TSD, Compound Word, Octuple) e justifique a escolha em função da tarefa. É permitido usar bibliotecas como `miditok`. Reporte o tamanho do vocabulário e o comprimento médio das sequências. Em tarefas condicionadas, descreva como a condição é representada (*tokens* de controle, *embedding* de atributo, sequência de entrada de um *encoder* etc.).

4. **Modelo.** Treine um modelo neural autoregressivo (gera um *token* por vez, condicionado aos *tokens* anteriores e, se houver, à condição).
   - (a) *Arquitetura:* deve ser escolhida entre **GRU**, **LSTM** ou **Transformer**.
   - (b) *Condicionamento:* o modelo pode ser condicionado ou não, dependendo da tarefa e do dataset (ex.: geração livre, *text2music*, arranjo automático).
   - (c) *Implementação:* o modelo deve ser implementado **nativamente em PyTorch**, sem o auxílio de bibliotecas de alto nível como Hugging Face (`transformers`, `Trainer`) ou `x-transformers`. Não é necessário implementar cada camada individualmente: podem ser usadas as camadas do PyTorch (`nn.Embedding`, `nn.GRU`, `nn.LSTM`, `nn.TransformerEncoderLayer`, `nn.MultiheadAttention` etc.). A classe do modelo, o laço de treinamento e o procedimento de amostragem (ex.: temperatura, *top-k*, *top-p*) devem ser escritos pelo grupo.

5. **Baseline.** Compare o modelo com pelo menos um *baseline* **mais simples** que ele. Por exemplo, se o modelo escolhido for um Transformer ou uma LSTM, o *baseline* pode ser uma RNN *vanilla* ou uma cadeia de Markov (como a do TP1). O *baseline* pode ser implementado com bibliotecas de alto nível ou com código de terceiros (devidamente citado), mas deve usar os mesmos dados, a mesma representação e o mesmo protocolo de avaliação do modelo principal.

6. **Experimentos.** Avalie as músicas geradas com pelo menos **3 métricas objetivas** de geração simbólica, comparando-as com as **músicas humanas do conjunto de teste**. É permitido usar bibliotecas como `muspy` (ex.: *pitch range*, *pitch class entropy*, *scale consistency*, *polyphony*, *empty beat rate*, *groove consistency*). Gere um número razoável de amostras por sistema (ex.: ≥ 100) e compare as **distribuições** das métricas, e não apenas as médias (ex.: distância para a média do conjunto de teste, *overlap area* ou divergência KL, como em Yang & Lerch, 2020).

7. **Ferramentas sugeridas.** `torch`, `miditok`, `muspy`, `pretty_midi`, `music21`. Para renderizar áudio a partir do MIDI gerado (apenas para audição), sugere-se `FluidSynth`/`pyFluidSynth` com um soundfont livre, ou MuseScore. Recomenda-se começar com um subconjunto do dataset e um modelo pequeno, garantindo que todo o pipeline (tokenizar, treinar, gerar, decodificar e avaliar) funcione antes de escalar.

8. **Formato do artigo ISMIR.** O artigo deve seguir o template oficial do ISMIR (disponível em ismir.net, LaTeX ou Word), com no máximo **4 páginas** de conteúdo (referências não contam no limite; ver exceção para pós-graduação em "Submissão"). Estrutura sugerida:
   - (a) Introdução – motivação e tarefa escolhida.
   - (b) Trabalhos relacionados – modelos e representações relevantes para a tarefa.
   - (c) Método – dataset e pré-processamento, representação, arquitetura do modelo e do *baseline*, condicionamento e amostragem.
   - (d) Experimentos – configuração experimental (partições, hiperparâmetros, hardware), métricas e protocolo de avaliação.
   - (e) Resultados e discussão – tabelas/figuras com as métricas, curvas de treino e validação, análise qualitativa das músicas geradas.
   - (f) Conclusão e trabalhos futuros – limitações e possíveis extensões.
   - (g) Referências.

## 3 Submissão

O trabalho deve ser feito em **grupos de 3 a 5 alunos**. A submissão deverá ser feita por um integrante do grupo na tarefa "Trabalho Final: Geração Musical Simbólica com Redes Neurais" no Moodle, contendo os seguintes entregáveis:

1. **Código-fonte no GitHub** – url para um repositório público contendo: código-fonte (pré-processamento, treino, geração e avaliação); `README.md` com instruções de instalação e reprodução dos resultados; arquivo de dependências (`requirements.txt`); e a descrição da contribuição de cada integrante do grupo.

2. **Artigo em formato ISMIR** – documento em pdf até 4 páginas de conteúdo (+ referências, sem limite), conforme descrito na seção de Instruções.

3. **Músicas geradas** – pelo menos 5 arquivos MIDI gerados pelo modelo e 5 pelo *baseline*, com os respectivos áudios sintetizados, incluídos no repositório e referenciados no artigo.

4. **Apresentação** – apresentação do trabalho em sala pelo grupo, nas aulas de Apresentação dos Trabalhos Finais.

5. **Adicional para grupos com alunos de pós-graduação:** grupos com pelo menos um aluno de pós-graduação devem escrever uma seção de *Trabalhos Relacionados* mais rigorosa, contendo a citação e discussão de pelo menos 5 artigos científicos recentes (últimos 8 anos) sobre geração musical simbólica com redes neurais. Esses grupos também devem realizar um **experimento adicional**. Sugestões:
   - (a) Métricas adicionais (ex.: perplexidade no conjunto de teste, memorização em relação ao conjunto de treino);
   - (b) Estudo com usuários preliminar (ex.: teste de audição comparando modelo, *baseline* e músicas humanas, com descrição dos participantes e do protocolo);
   - (c) Ablações (tamanho do modelo, comprimento do contexto, representação) ou estudo de estratégias de amostragem (temperatura, *top-k*, *top-p*).

   Por conta desses requisitos extras, o artigo desses grupos pode ter até **6 páginas** de conteúdo (em vez de 4).
