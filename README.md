# QA sobre manual técnico com LLM local

Trabalho da disciplina IC0024 (PGCOMP/UFBA). Sistema de perguntas e respostas (QA) em português sobre um manual
confidencial de uma empresa do setor portuário, com arquitetura Retriever-Reader e leitor generativo (RAG). O
leitor é ajustado com QLoRA, e todos os modelos são abertos e servidos pelo [Ollama](https://ollama.com).

Dada uma pergunta, o sistema recupera 5 trechos do manual (BM25, busca densa ou híbrida) e o leitor responde em uma
ou duas frases, terminando com a fonte no formato `(seção X, p. N)`. Quando a informação não está nos trechos, a
resposta é `Não encontrado no manual.`

## Declaração de confidencialidade

**Nenhum dado do manual sai da máquina local.**

- A única rede usada em tempo de execução é o Ollama em `http://localhost:11434`. O cliente
  ([qa_manual/ollama_client.py](qa_manual/ollama_client.py)) recusa endereços que não sejam locais. Downloads de
  modelos e pacotes acontecem só na instalação.
- O Google Colab recebe **apenas o documento público** usado no ajuste fino (um regulamento portuário publicado
  por uma autoridade portuária). O notebook recusa nomes de arquivo do manual e exige confirmação explícita de
  que o PDF é público.
- Logs (`runs/<run_id>/log.txt`) registram ids, contagens, tempos e hashes, nunca texto. O cache do BERTScore
  guarda só hashes.
- `data/`, `index/`, `runs/`, `models/` e `*.gguf` estão no [.gitignore](.gitignore). O repositório contém só
  código, prompts, configurações, testes e documentação. Testes e exemplos usam texto inventado ou o manual
  público de [inputs_teste/](inputs_teste/).
- **Pastas sincronizadas com a nuvem também são "fora da máquina".** Se o projeto estiver na Mesa ou em
  Documentos com o iCloud ligado, ou em Dropbox, OneDrive ou Google Drive, o manual e os índices seriam enviados
  para a nuvem. `qa-manual indexar` detecta essas pastas e se recusa a gravar nelas; `qa-manual smoke` avisa.
  Mantenha o projeto (e o ambiente virtual) numa pasta local, como `~/projetos/qa-manual`. No iCloud, uma pasta
  terminada em `.nosync` também fica fora da sincronização.

## Pipeline

| Fase | Passos | Onde roda | Quando |
|---|---|---|---|
| Indexação | 1 a 4 | local, CPU | uma vez por manual |
| Ajuste do leitor | 5 a 7 | Google Colab (T4), com documento público | uma vez |
| Consulta | 8 a 11 | local, CPU | a cada pergunta |
| Avaliação | 12 a 15 | local, CPU | ao final |

```
manual.pdf
  -> [1] PyMuPDF: texto por página           -> [2] blocos + regex de títulos -> parágrafos com tema e página
  -> [3] trechos de até ~400 tokens por tema -> data/trechos.jsonl
  -> [4a] bm25s -> index/bm25/               -> [4b] Ollama bge-m3 -> FAISS IndexFlatIP -> index/dense.faiss

documento_publico.pdf (Colab)
  -> passos 1 a 3 -> [5] llama3.2:3b gera pares -> [6] Unsloth QLoRA sobre Qwen3-4B
  -> [7] GGUF q4_k_m + Modelfile -> leitor "qwen3-manual:4b" no Ollama local

pergunta
  -> [8] BM25 e bge-m3/FAISS -> [9] RRF -> top-5 -> [10] leitor (base ou ajustado) -> [11] verificador
  -> runs/<run_id>/respostas.jsonl

trechos -> [12] candidatas (llama3.2:3b) + revisão humana -> gold_dev / gold_test
        -> [13] EM, F1, BERTScore, Recall@5, MRR, abstenção -> [14] perplexidade (llama.cpp)
        -> [15] metrics.csv, pareado.csv e gráficos
```

Configurações experimentais: **S0** (sem recuperação), **S1** (BM25), **S2** (denso), **S3** (híbrido RRF). Cada
uma roda com dois leitores: `base` (`qwen3:4b`) e `ajustado` (`qwen3-manual:4b`). Só o recuperador e o leitor
mudam; prompt, `k` e decodificação (temperatura 0, seed 42) são idênticos.

## Instalação local

Requisitos: Python 3.11, 8 a 16 GB de RAM, sem GPU, [Ollama](https://ollama.com/download) instalado e rodando
(`ollama serve`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -c "import nltk; nltk.download('stopwords')"
ollama pull qwen3:4b && ollama pull bge-m3 && ollama pull llama3.2:3b
qa-manual smoke
```

- `requirements.txt` fixa as versões usadas no desenvolvimento. Em Linux sem GPU, instale antes o torch para
  CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- O BERTimbau (`neuralmind/bert-base-portuguese-cased`, ~430 MB) é baixado do Hugging Face na primeira execução
  do BERTScore. É download de modelo, não envio de dados.
- No macOS, `faiss-cpu` e `torch` trazem cópias diferentes do OpenMP e não podem ser carregados no mesmo
  processo. Por isso o BERTScore roda num subprocesso ([qa_manual/bertscore_proc.py](qa_manual/bertscore_proc.py)).

### llama.cpp (só o binário de perplexidade)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=OFF && cmake --build build --target llama-perplexity -j
# binário em build/bin/llama-perplexity
```

## Ordem de execução do experimento

```bash
qa-manual indexar --manual data/manual.pdf
qa-manual stats                                       # confira temas detectados e tamanhos dos trechos
qa-manual gerar-perguntas --n 150                     # -> data/gold_candidatas.jsonl
qa-manual exportar-revisao                            # -> data/gold_revisao.csv (revisão humana)
qa-manual importar-revisao --csv data/gold_revisao.csv --dividir --reservar-ppl 30
# (Colab) gerar o GGUF e criar qwen3-manual:4b localmente; veja "Ajuste fino no Colab"
qa-manual perplexidade --binario ~/llama.cpp/build/bin/llama-perplexity \
    --gguf-base ollama --gguf-ajustado models/qwen3-manual-q4_k_m.gguf
qa-manual avaliar --gold data/gold_dev.jsonl --configs S3 --leitores base        # ajuste de k e max_tokens
qa-manual avaliar --gold data/gold_test.jsonl --configs S0 S1 S2 S3 --leitores base ajustado
```

**Experimento mínimo viável** (antes de qualquer refinamento): indexar, gerar e revisar o gold e rodar
`qa-manual avaliar --gold data/gold_test.jsonl --configs S0 S3 --leitores base --limite 30`.

A revisão humana segue o [protocolo de anotação](docs/protocolo_anotacao.md). O CSV usa `;` e UTF-8 com BOM e abre
direto no Excel em português e no LibreOffice. Preencha `decisao` (`aceitar`, `editar` ou `descartar`) e, ao
editar, `pergunta_corrigida` e/ou `resposta_corrigida`.

A reserva para perplexidade (`--reservar-ppl 30`) escolhe 30 trechos de temas que não estão no gold de teste. Ela
também existe em `indexar`, mas ali, antes da divisão, não há como excluir os temas do teste (o comando avisa).

### Comandos

Todos aceitam `--config configs/base.yaml` (padrão) e `--sobrescrever <yaml>`. Arquivo de entrada omitido é
pedido interativamente; arquivo inexistente encerra com código 2 e uma mensagem que diz o que fazer.

| Comando | Passos | Saídas |
|---|---|---|
| `indexar --manual <pdf> [--reservar-ppl N] [--max-tokens N] [--sobreposicao-paragrafos N]` | 1–4 | `data/trechos.jsonl`, `data/trechos_stats.json`, `index/bm25/`, `index/bm25_ids.json`, `index/dense.faiss`, `index/dense_ids.json`, `index/manifest.json` |
| `recuperar --pergunta "..." --modo bm25\|denso\|hibrido [--k 5]` | 8–9 | ranking no terminal |
| `perguntar --config-exp S3 --leitor base\|ajustado --pergunta "..."` ou `--arquivo perguntas.txt` (sem os dois, modo interativo) | 8–11 | resposta no terminal; `runs/<data>_S3_base/respostas.jsonl`, `config.json`, `log.txt` |
| `gerar-perguntas [--n 150]` | 12 | `data/gold_candidatas.jsonl` |
| `exportar-revisao` | 12 | `data/gold_revisao.csv` (não sobrescreve sem `--forcar`) |
| `importar-revisao --csv <csv> [--dividir] [--reservar-ppl N]` | 12 | `data/gold_revisado.jsonl`, `data/gold_dev.jsonl`, `data/gold_test.jsonl`, `data/ppl_holdout.txt` |
| `perplexidade --binario <llama-perplexity> --gguf-base <gguf\|ollama> --gguf-ajustado <gguf>` | 14 | `runs/<data>_perplexidade/perplexidade.json` |
| `avaliar --gold <jsonl> --configs S0 S1 S2 S3 --leitores base ajustado [--repeticoes 1] [--retomar <run_id>] [--limite N]` | 13, 15 | `runs/<data>_avaliar/`: `respostas.jsonl`, `metrics.csv`, `pareado.csv`, `respostas_por_config.png`, `recall_por_config.png`, `config.json`, `log.txt` |
| `smoke [--sem-rede] [--sem-bertscore]` | — | versões e testes de Ollama (chat e embed), FAISS, bm25s e BERTScore |
| `stats` | — | `trechos_stats.json` e `manifest.json` |

`perguntar` e `avaliar` recusam rodar se o índice não existe ou se o PDF, os trechos, o modelo de embeddings ou os
parâmetros do BM25 mudaram desde a indexação (`índice desatualizado; rode indexar`). `avaliar` grava cada resposta
assim que ela sai; se for interrompido, `--retomar <run_id>` continua do ponto em que parou. Para medir latência
sem o tempo de carga, cada leitor é aquecido antes da medição.

`--gguf-base ollama` lê a linha `FROM` de `ollama show qwen3:4b --modelfile`: o blob apontado
(`~/.ollama/models/blobs/sha256-...`) é um GGUF válido para o llama.cpp. `avaliar` inclui a perplexidade do
`perplexidade.json` mais recente em `runs/` (ou do arquivo indicado em `--perplexidade`).

## Ajuste fino no Colab

[colab/finetune_qlora.ipynb](colab/finetune_qlora.ipynb) executa os passos 5 a 7 numa GPU T4:

1. Instala Unsloth e Ollama, clona este repositório (ajuste `REPO_URL`) e roda os passos 1 a 3 sobre o documento
   público, com o mesmo código e os mesmos parâmetros do projeto.
2. **Passo 5:** o `llama3.2:3b` gera pergunta e resposta por trecho (até 1500). A resposta termina com a fonte do
   próprio trecho, preenchida pelo código; 20% dos exemplos têm como alvo `Não encontrado no manual.`. Cada
   exemplo usa o prompt do leitor ([prompts/leitor.txt](prompts/leitor.txt)) com o trecho-evidência e 2 a 4
   distratores (BM25), em ordem aleatória. 5% vão para validação.
3. **Passo 6:** QLoRA sobre `unsloth/Qwen3-4B` em 4 bits (`r=16`, `alpha=16`, `dropout=0`, 7 projeções), batch 2 ×
   acumulação 8, `lr=2e-4`, 2 épocas, cosseno com aquecimento de 3%, `adamw_8bit`, fp16 na T4, seed 42. Chat
   template do Qwen3 com `enable_thinking=False` e perda só nos tokens da resposta. Exemplos com mais de 2048
   tokens são descartados (truncados, perderiam a resposta), por isso o contexto de treino fica limitado a ~1400
   tokens de trechos.
4. Checagem de sanidade em 10 exemplos de validação (formato da citação e abstenção).
5. **Passo 7:** `save_pretrained_gguf(..., quantization_method="q4_k_m")` e download de
   `qwen3-manual-q4_k_m.gguf`, `treino_log.csv` e `metricas_validacao.json`.

Na máquina local:

```bash
mkdir -p models && mv ~/Downloads/qwen3-manual-q4_k_m.gguf models/
mv ~/Downloads/treino_log.csv ~/Downloads/metricas_validacao.json colab/
bash colab/criar_modelo_local.sh models/qwen3-manual-q4_k_m.gguf
```

O script gera o `Modelfile` a partir de `ollama show qwen3:4b --modelfile`, trocando só a linha `FROM` (TEMPLATE,
PARAMETER e SYSTEM ficam iguais; veja [colab/Modelfile.template](colab/Modelfile.template)), roda
`ollama create qwen3-manual:4b -f Modelfile` e testa com `ollama run`.

## Configuração

[configs/base.yaml](configs/base.yaml) concentra todos os parâmetros: caminhos, modelos, ingestão, trechos, BM25,
denso, recuperação (`k=5`, 20 candidatos por lista, `rrf_c=60`), decodificação (`temperature 0`, `seed 42`,
`num_ctx 8192`, `num_predict 200`, `think false`), abstenção, gold e avaliação. `S0.yaml` a `S3.yaml` só trocam
`retriever` (`nenhum`, `bm25`, `denso`, `hibrido`).

As sobrescritas são mescladas seção a seção: um YAML com `trechos: {max_tokens: 300}` muda só esse valor. Chave
desconhecida, tipo errado ou valor fora do domínio encerra com erro. `OLLAMA_HOST` substitui `ollama.host`, que
continua restrito a endereços locais.

## Contratos de dados

| Arquivo | Campos |
|---|---|
| `data/trechos.jsonl` | `id` (`t0001`...), `tema`, `tema_id`, `pagina_inicio`, `pagina_fim`, `paragrafos`, `n_tokens`, `texto` |
| `index/manifest.json` | `criado_em`, `manual_pdf`, `hash_pdf_sha256`, `hash_trechos_sha256`, `n_trechos`, `n_paginas`, `modelo_embeddings`, `dimensao`, `params_trechos`, `params_ingestao`, `bm25`, `versao_codigo` |
| `data/gold_candidatas.jsonl` | `id`, `pergunta`, `resposta_ref`, `evidencia`, `tipo` (`factual`, `procedimental`, `sem_resposta`), `tema_id`, `origem`, `revisado`, `trecho_origem` |
| `data/gold_revisao.csv` | `id`, `tipo`, `tema_id`, `evidencia`, `pergunta`, `resposta_ref`, `texto_evidencia`, `decisao`, `pergunta_corrigida`, `resposta_corrigida`, `observacao` |
| `data/gold_dev.jsonl`, `data/gold_test.jsonl` | campos das candidatas com `revisado: true` e `split`; todos os itens de um tema caem no mesmo split |
| `runs/<run_id>/respostas.jsonl` | `run_id`, `config`, `retriever`, `leitor`, `modelo`, `q_id`, `pergunta`, `tipo`, `recuperadas` (`id`, `score`, `rank`), `resposta`, `absteve`, `citou_fonte`, `fonte_valida`, `citacoes`, `latencia_s`, `t_embedding_s`, `t_busca_s`, `t_leitura_s`, `n_tokens_prompt`, `n_tokens_resposta` (e `repeticao` na avaliação) |
| `runs/<run_id>/metrics.csv` | `config`, `leitor`, `n`, `em`, `f1_tokens`, `bertscore_f1`, `recall_at_5`, `mrr_at_5`, `abst_correta`, `abst_indevida`, `alucinacao_sem_resposta`, `fonte_valida_pct`, `latencia_mediana_s`, `latencia_p95_s`, `perplexidade` |

## Métricas

- **Respondíveis** (`factual`, `procedimental`): EM e F1 de tokens (estilo SQuAD) após normalização (minúsculas,
  sem pontuação, sem artigos e preposições frequentes, acentos mantidos) e BERTScore F1 com o BERTimbau (9
  camadas, sem reescala). A citação `(seção X, p. N)` é removida antes da comparação. Se o sistema se absteve, EM,
  F1 e BERTScore valem 0. Recall@5 e MRR@5 sobre os trechos recuperados (NaN em S0).
- **Sem resposta:** `abst_correta` é a fração em que o sistema se absteve; `alucinacao_sem_resposta = 1 -
  abst_correta`. `abst_indevida` é a fração de respondíveis em que ele se absteve.
- **Fonte válida:** entre as respostas não abstidas, fração cuja citação coincide com a seção e a página de um
  trecho recuperado.
- **Eficiência:** latência mediana e p95 por resposta.
- **Perplexidade:** `llama-perplexity` sobre os trechos reservados, leitor base × ajustado.
- **Ganho pareado** (`pareado.csv`): diferença ajustado − base em EM e F1, pergunta a pergunta, com intervalo de
  confiança de 95% por bootstrap (1000 reamostras).

## Rede desligada

Depois dos `ollama pull` e da primeira execução do BERTScore, `indexar`, `perguntar`, `gerar-perguntas`,
`perplexidade` e `avaliar` funcionam sem rede. `qa-manual smoke --sem-rede` confirma que a rede externa está
inacessível.

## Testes

```bash
pip install -e ".[dev]"
pytest --cov          # sem Ollama real e sem o manual; cobertura mínima de 80% (exceto cli.py e perplexity.py)
ruff check . && ruff format --check .
```

Os testes usam um PDF sintético de 6 páginas gerado em [tests/conftest.py](tests/conftest.py) (cabeçalho
repetido, títulos `1 Introdução`, `2.1 Atracação`, `2.2 Liberação de carga`, um parágrafo de 700 palavras e
hifenização) e o `OllamaFake` (embeddings por hash, leitor extrativo). Um teste garante que nenhum texto de trecho
aparece no `log.txt`.

[inputs_teste/](inputs_teste/) traz um manual público do Porto de Salvador e um gold para testar tudo de ponta a
ponta com o Ollama real, sem o manual da empresa. Veja [inputs_teste/README.md](inputs_teste/README.md).

## Estrutura

```
configs/      base.yaml e S0-S3
prompts/      leitor, leitor sem trechos (S0), geradores de perguntas e de pares de treino
qa_manual/
  config.py         YAML -> Config (dataclasses) com validação
  io_utils.py       JSONL, hashes, run_id, log de execução, detecção de pasta sincronizada
  ingest.py         passos 1-2: blocos, cabeçalhos/rodapés, hifenização, títulos e temas
  chunking.py       passo 3: trechos por tema
  tokenize_pt.py    tokenização e stop-words (NLTK) para o BM25
  index_bm25.py     passo 4a          index_dense.py   passo 4b
  indexacao.py      orquestra 1-4, manifesto e carga dos índices
  ollama_client.py  cliente real (só localhost) e falso
  retrieve.py       passos 8-9 (BM25, denso, RRF)
  prompts.py        templates          generate.py      passo 10
  verify.py         passo 11           pipeline.py      orquestra 8-11
  gold.py           passo 12           metrics.py       passo 13
  perplexity.py     passo 14           evaluate.py      passo 15
  bertscore_proc.py BERTScore em subprocesso
  cli.py            comandos
colab/        notebook do ajuste fino, Modelfile.template, criar_modelo_local.sh
docs/         protocolo de anotação, modelos de relatório e slides
inputs_teste/ manual público de teste, gold e casos de erro
tests/        pytest (PDF sintético e Ollama falso)
```

## Decisões de implementação

Pontos em que o código detalha ou ajusta o plano original:

- **Títulos fora de sequência viram texto** (`ingestao.validar_sequencia_titulos: true`). A regex de títulos
  também casa com passos numerados (`1 Acionar o alarme...`) e linhas de tabela (`202 Armazéns 3 e 4...`). Um
  título só é aceito se a numeração puder suceder a seção atual (avançar até 5 em algum nível ou abrir subseção).
  Linhas de sumário com pontilhado são descartadas.
- **Mescla de configuração seção a seção**, em vez de rasa, para permitir sobrescrever um único parâmetro.
- **`llama-perplexity` sem `--chunks 0`:** no llama.cpp, `--chunks 0` avaliaria zero blocos; o padrão (`-1`)
  avalia o texto inteiro.
- **Citação removida antes de EM/F1/BERTScore**, senão nenhuma resposta do leitor seria igual à referência.
- **Novas tentativas do gerador mudam a seed:** com temperatura 0 e a mesma seed, a segunda tentativa repetiria a
  primeira. O gerador também usa `format="json"` do Ollama.
- **Proteção contra pastas sincronizadas** em `indexar` (veja a declaração de confidencialidade).
- `respostas.jsonl` tem o campo extra `retriever`, e as candidatas têm `trecho_origem` (o trecho que originou a
  pergunta), usado para mostrar o texto ao revisor também nos itens `sem_resposta`.

## Referências

- Cortes, Vieira e Barone (2024). Perguntas e Respostas. Cap. 16 de *Processamento de Linguagem Natural:
  Conceitos, Técnicas e Aplicações em Português*, 2ª ed. (BPLN).
- Zhang et al. (2023). A Survey for Efficient Open Domain Question Answering. ACL.
- Perera et al. (2025). A Survey of the State-of-the-Art in Conversational Question Answering Systems.
  arXiv:2509.05716.
- Zaib et al. (2022). Conversational question answering: a survey. *Knowledge and Information Systems*.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
- Robertson e Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in
  Information Retrieval*.
- Cormack, Clarke e Büttcher (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning
  Methods. SIGIR.
- Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
