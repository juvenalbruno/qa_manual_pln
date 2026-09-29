# QA sobre manual técnico com LLM local (Ollama)

Trabalho da disciplina IC0024 (PGCOMP/UFBA): question answering em domínio fechado, open-book e generativo.
O sistema recebe um manual em PDF e uma pergunta em português, e devolve uma resposta curta com a seção e a
página de origem. Quando a informação não está no documento, responde "Não encontrado no manual".

Arquitetura retriever-reader: BM25 (`bm25s`) e busca densa (`bge-m3` + FAISS) combinados por Reciprocal Rank
Fusion; o leitor é o `qwen3:8b` e o juiz automático é o `llama3.1:8b`, todos servidos pelo Ollama.

## Declaração de isolamento dos dados

**Nenhum dado do manual sai da máquina.** Todo o processamento (extração, embeddings, geração, julgamento) roda
localmente:

- O cliente do Ollama ([src/ollama_client.py](src/ollama_client.py)) **recusa URLs que não sejam locais**
  (`localhost`, `127.0.0.1`, `::1`).
- Não há chamadas a APIs externas. Depois de instalar as dependências e baixar os modelos, todos os comandos
  funcionam com a rede desligada.
- `data/`, `index/`, `runs/` e arquivos `*.pdf` estão no [.gitignore](.gitignore). O repositório contém apenas
  código, prompts, configurações e estatísticas agregadas.
- Os exemplos citados no relatório e nos slides são parafraseados.

## Instalação

```bash
# 1. Ollama (https://ollama.com/download) e modelos
ollama pull qwen3:8b        # leitor
ollama pull bge-m3          # embeddings
ollama pull llama3.1:8b     # juiz e gerador de perguntas (distinto do leitor)

# 2. Ambiente Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Teste de fumaça (repita com a rede desligada para o critério de aceite da etapa 0)
python smoke_test.py --gravar-readme
```

## Ambiente

Preenchido por `python smoke_test.py --gravar-readme`:

<!-- AMBIENTE:INICIO -->
_(execute o teste de fumaça para registrar hardware, versões do Ollama, tags dos modelos e bibliotecas)_
<!-- AMBIENTE:FIM -->

## Inputs de teste

[inputs_teste/](inputs_teste/) traz um manual público do Porto de Salvador em PDF, 152 perguntas de gold
(dev e teste), consultas de teste, perguntas avulsas e casos de erro, para testar tudo sem o manual da empresa.
Veja o roteiro em [inputs_teste/README.md](inputs_teste/README.md).

## Execução com um único comando

```bash
python -m src.cli executar \
  --manual     data/manual.pdf \
  --gold       data/gold_test.jsonl \
  --gold-dev   data/gold_dev.jsonl \
  --consultas  data/consultas_teste.jsonl \
  --perguntas  perguntas.txt \
  --configs S0 S1 S2 S3 --repeticoes 3
```

O comando valida as entradas, confere o Ollama, indexa (ou reaproveita o índice), testa o índice, responde as
perguntas avulsas, avalia e analisa. Com `--varrer`, ajusta tamanho de passagem e k no dev antes. Só `--manual` e
`--gold` são obrigatórios. Veja `python -m src.cli executar --help`.

## Reprodução, etapa por etapa

Todos os comandos rodam da raiz do repositório. **Argumentos de arquivo omitidos são pedidos interativamente**;
nenhum caminho de dados está fixo no código. Arquivos inexistentes ou com extensão errada encerram o comando com
código 2 e uma mensagem clara.

```bash
# Etapas 1 e 2: extrair, limpar, segmentar e indexar
python -m src.cli indexar --manual data/manual.pdf
python -m src.cli inspecionar --n 20                      # inspeção manual de 20 passagens
python -m src.cli testar-indice --consultas data/consultas_teste.jsonl   # 5 consultas à mão (modelo em exemplos/)

# Etapa 3: depuração da recuperação
python -m src.cli recuperar --pergunta "Qual o prazo para liberação da carga?" --modo hibrido

# Etapa 4: perguntas avulsas (sem argumentos entra em modo interativo)
python -m src.cli perguntar --config S3
python -m src.cli perguntar --config S3 --arquivo perguntas.txt

# Etapa 5: conjunto de avaliação
python -m src.cli gerar-perguntas --n 150
python -m src.cli revisar --candidatas data/gold_candidatas.jsonl --anotador ana
python -m src.cli revisar --candidatas data/gold_candidatas.jsonl --anotador beto --comum   # 30% em comum
python -m src.cli concordancia --revisoes data/revisoes/ana.jsonl data/revisoes/beto.jsonl
python -m src.cli dividir-gold --candidatas data/gold_candidatas.jsonl --revisoes data/revisoes/ana.jsonl data/revisoes/beto.jsonl

# Etapa 6: avaliação
python -m src.cli avaliar --gold data/gold_test.jsonl --configs S0 S3 --limite 30          # experimento mínimo
python -m src.cli avaliar --gold data/gold_test.jsonl --configs S0 S1 S2 S3 --repeticoes 3
python -m src.cli validar-juiz --run runs/<timestamp>_avaliar                              # 50 itens à mão

# Etapa 7: ajuste no dev, análise no teste
python -m src.cli varrer --gold data/gold_dev.jsonl --manual data/manual.pdf --tamanhos 300 400 500 --ks 3 5 8
python -m src.cli analisar --run runs/<timestamp>_avaliar
python -m src.cli classificar-erros --run runs/<timestamp>_avaliar
python -m src.cli analisar --run runs/<timestamp>_avaliar     # de novo, com as categorias confirmadas
```

`avaliar` e `varrer` podem ser retomados após uma interrupção com `--retomar runs/<timestamp>_...`: os itens já
executados não são repetidos. `avaliar` primeiro gera todas as respostas e depois roda o juiz, para não alternar
dois modelos de 8B na memória.

### O que cada comando grava

| Comando | Saída |
|---|---|
| `indexar` | `index/passages.jsonl`, `stats.json`, `bm25/`, `dense.faiss`, `dense_ids.json`, `manifest.json` |
| `perguntar` | `runs/<ts>_perguntar_<cfg>/respostas.jsonl` |
| `gerar-perguntas` | `data/gold_candidatas.jsonl` |
| `revisar` / `concordancia` | `data/revisoes/<anotador>.jsonl`, `data/revisoes/concordancia.json` |
| `dividir-gold` | `data/gold_dev.jsonl`, `data/gold_test.jsonl` |
| `avaliar` | `runs/<ts>_avaliar/respostas.jsonl`, `julgamentos.jsonl`, `metrics.json` + tabela no terminal |
| `validar-juiz` | `runs/<ts>/juiz_manual.jsonl` e `_validacao_juiz` em `metrics.json` |
| `varrer` | `index/t<tamanho>/`, `runs/<ts>_varredura/varredura.json` |
| `analisar` | `runs/<ts>/analise.md`, `comparacoes.json`, `recall_k.csv`, `recall_k.png`, `erros_amostra.jsonl` |
| `classificar-erros` | `runs/<ts>/erros_classificados.jsonl` |

`perguntar` e `avaliar` conferem `index/manifest.json`: se o índice não existe, se o manual mudou (hash SHA-256)
ou se o modelo de embeddings difere da configuração, pedem para rodar `indexar` antes.

## Configurações

| Arquivo | Conteúdo |
|---|---|
| [configs/base.yaml](configs/base.yaml) | parâmetros comuns: modelos, `k`, RRF, temperatura 0, seed 42, `num_ctx` 8192, `num_predict` 200, `think: false` |
| [configs/indexacao.yaml](configs/indexacao.yaml) | passagem de 400 tokens (~300 palavras), sobreposição de 50 tokens |
| [configs/S0.yaml](configs/S0.yaml) | closed-book (sem passagens) |
| [configs/S1.yaml](configs/S1.yaml) / [S2](configs/S2.yaml) / [S3](configs/S3.yaml) | BM25 / denso / híbrido RRF |
| [configs/S3_gemma3_4b.yaml](configs/S3_gemma3_4b.yaml) | opcional: S3 com um leitor menor |
| [configs/S3_qwen3_4b.yaml](configs/S3_qwen3_4b.yaml) | S3 com `qwen3:4b`, para iterar rápido no desenvolvimento |

`OLLAMA_HOST` (a mesma variável do Ollama) sobrescreve a URL, que continua restrita a endereços locais.

## Métricas

- **Recuperação:** Recall@k (acerto se alguma passagem-evidência está no top-k) e MRR.
- **Resposta:** Exact Match e F1 de tokens após normalização (minúsculas, sem pontuação, sem artigos e preposições
  frequentes). A citação "(seção X, p. N)" é removida antes da comparação. Itens `sem_resposta` valem 1 quando o
  sistema se abstém. `em_respondiveis` e `f1_respondiveis` excluem esses itens.
- **Juiz local:** nota 0/1/2 do `llama3.1:8b` e fidelidade 0/1 (a resposta é sustentada pelas passagens?).
  Abstenções são pontuadas sem chamar o modelo.
- **Abstenção:** correta (itens `sem_resposta`) e indevida (demais itens).
- **Citação correta:** a página citada pertence à passagem-evidência.
- **Eficiência:** latência mediana e p95. O modelo é pré-carregado antes da medição.
- **Reprodutibilidade:** fração de perguntas com resposta idêntica entre repetições.

## Testes

```bash
pytest -q
```

Os testes não precisam do Ollama: [tests/fake_ollama.py](tests/fake_ollama.py) sobe um servidor local que imita
`/api/chat` e `/api/embed`, e [scripts/gerar_manual_exemplo.py](scripts/gerar_manual_exemplo.py) gera um manual
fictício com cabeçalhos, rodapés, tabela e hifenização. O teste ponta a ponta executa todos os comandos da CLI.

## Estrutura

```
configs/     S0-S3 e parâmetros de indexação
prompts/     leitor, leitor sem contexto (S0), gerador de perguntas, juiz e juiz de fidelidade
src/
  cli.py            comandos e solicitação dos arquivos de entrada
  ingest.py         extração (PyMuPDF), limpeza, títulos, tabelas, segmentação
  index.py          índices BM25 e FAISS, manifesto
  retrieve.py       BM25, denso, híbrido (RRF)
  ollama_client.py  /api/chat e /api/embed (somente localhost)
  generate.py       prompt do leitor, geração, extração da citação
  evaluate.py       métricas, juiz, agregação, orquestração da avaliação
  gold.py           geração de candidatas, revisão, concordância, divisão por seção
  analysis.py       bootstrap, McNemar, Recall@k, análise de erros, varredura
  utils.py          normalização, JSONL, configurações
docs/        protocolo de anotação, modelos de relatório e slides
exemplos/    formatos de consultas de teste e de arquivo de perguntas
scripts/     gerador do manual fictício usado nos testes
tests/       testes de unidade e ponta a ponta
```

## Limitações conhecidas

- A detecção de títulos é heurística (numeração, caixa alta, tamanho de fonte). Manuais sem hierarquia visual
  clara podem gerar seções erradas; verifique com `inspecionar`.
- Tabelas complexas (células mescladas) podem ser mal extraídas pelo PyMuPDF. Se forem relevantes, troque por
  Docling.
- Em CPU, cada pergunta leva de 10 a 30 s com modelos de 8B. Com 8 GB de RAM, prefira `qwen3:4b` no
  desenvolvimento (`--configs S3_qwen3_4b`) e rode o 8B só no teste final.
