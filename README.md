# QA sobre o manual

Sistema de perguntas e respostas sobre um manual técnico do setor portuário, feito para a disciplina IC0024
(PGCOMP/UFBA). Ele lê o `manual.pdf`, busca os trechos mais relevantes para cada pergunta e pede a um modelo de
linguagem local (Ollama) que responda usando só esses trechos, citando seção e página.

Tudo roda na sua máquina: o manual não é enviado para nenhum serviço externo.

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `qa_manual.py` | O sistema de QA: indexa o manual, responde perguntas e gera respostas para revisão |
| `avaliar.py` | Métricas (separado do sistema principal) |
| `manual.pdf` | O manual (você coloca aqui; não vai para o git) |
| `perguntas.json` | As perguntas de avaliação (você escreve; não vai para o git) |

## Instalação (macOS)

```bash
# 1. Python 3.11 (com Homebrew)
brew install python@3.11

# 2. Ambiente virtual e bibliotecas
python3.11 -m venv .venv
source .venv/bin/activate
pip install pymupdf==1.28.2 bm25s==0.3.11 numpy==2.4.6 ollama==0.6.3 matplotlib==3.11.2

# 3. Ollama (https://ollama.com/download) e os modelos
brew install ollama          # ou baixe o aplicativo no site
ollama serve                 # em outro terminal (o aplicativo já faz isso sozinho)
ollama pull qwen3:4b         # leitor: escreve as respostas
ollama pull bge-m3           # embeddings: busca por significado
ollama pull llama3.2:3b      # propõe as respostas do perguntas.json para revisão
```

Em cada terminal novo, ative o ambiente com `source .venv/bin/activate`. No Linux, troque o `brew` pelo gerenciador
de pacotes e instale o Ollama com `curl -fsSL https://ollama.com/install.sh | sh`.

Não deixe a pasta do projeto na Mesa ou em Documentos se o iCloud estiver sincronizando essas pastas (nem em
Dropbox, OneDrive ou Google Drive): o manual e o índice iriam para a nuvem.

## Uso

### 1. Indexar o manual (uma vez)

Coloque o `manual.pdf` ao lado do `qa_manual.py` e rode:

```bash
python qa_manual.py indexar
```

Cria a pasta `indice/`. Rode de novo só se o manual mudar.

### 2. Fazer perguntas

```bash
python qa_manual.py perguntar
```

```
Pergunta:
Qual o prazo para liberação da carga?
Resposta: A carga é liberada em até 48 horas após a atracação (seção 4.2, p. 17).

Pergunta:
```

Linha vazia ou `sair` encerra. Para uma pergunta só: `python qa_manual.py perguntar --pergunta "..."`.

### 3. Avaliar

**a) Escreva o `perguntas.json`** ao lado do `qa_manual.py` (de 50 a 100 perguntas; inclua algumas que o manual
não responde, para medir se o sistema sabe dizer que não sabe):

```json
[
  {"id": 1, "pergunta": "Qual o prazo para liberação da carga?"},
  {"id": 2, "pergunta": "Qual o salário do operador de guindaste?"}
]
```

**b) Gere as respostas propostas:**

```bash
python qa_manual.py responder
```

Cria o `resposta.json`. As respostas propostas são escritas pelo `llama3.2:3b`, que não é o leitor avaliado: assim o
leitor não é comparado com as próprias respostas.

**c) Revise o `resposta.json`.** Para cada item:

```json
{
  "id": "1",
  "tipo": "factual",
  "pergunta": "Qual o prazo para liberação da carga?",
  "resposta": "Até 48 horas após a atracação.",
  "secao": "4.2",
  "pagina": 17,
  "trecho": "t0042",
  "aprovada": false,
  "observacao": ""
}
```

- Resposta certa: troque `"aprovada"` para `true`.
- Resposta errada: corrija o texto de `"resposta"` (e `"secao"`/`"pagina"`, se for outro lugar) e marque `true`.
- Pergunta que o manual não responde: `"resposta": "Não encontrado no manual"`, `"tipo": "sem_resposta"`, `true`.
- Itens com `false` são ignorados na avaliação.

O `responder` não sobrescreve um `resposta.json` existente (use `--forcar` se quiser gerar de novo).

**d) Calcule as métricas:**

```bash
python avaliar.py                          # S0 a S3 com o leitor qwen3:4b
python avaliar.py --modos S0 S3 --limite 30  # versão rápida
```

Modos de busca comparados: **S0** sem busca (o modelo responde sozinho), **S1** BM25 (palavras), **S2** denso
(embeddings), **S3** híbrido (BM25 + denso fundidos por RRF). Para comparar outro leitor:
`python avaliar.py --modelos qwen3:4b outro-modelo`.

Os resultados ficam em `resultados/<data-hora>/`:

| Arquivo | Conteúdo |
|---|---|
| `metrics.csv` | Uma linha por modo e modelo |
| `respostas.jsonl` | Cada resposta, com os trechos buscados |
| `grafico.png` | EM, F1 e Recall@5 por modo |

| Métrica | O que mede |
|---|---|
| `em` | Resposta igual ao gabarito (após normalizar) |
| `f1` | Sobreposição de palavras com o gabarito (0 a 1) |
| `recall_at_5` | O trecho com a resposta está entre os 5 buscados |
| `mrr_at_5` | 1 / posição desse trecho na busca |
| `abst_correta` | Nas perguntas sem resposta, quantas vezes o sistema disse "Não encontrado no manual" |
| `abst_indevida` | Nas perguntas com resposta, quantas vezes ele disse "Não encontrado" sem necessidade |
| `citacao_valida` | A seção e a página citadas batem com um trecho buscado |
| `latencia_mediana_s` | Tempo mediano por pergunta |

## Problemas comuns

| Mensagem | O que fazer |
|---|---|
| `Ollama não respondeu` | Abra o aplicativo do Ollama ou rode `ollama serve` |
| `modelo ... não encontrado` | `ollama pull <modelo>` |
| `índice não encontrado` | `python qa_manual.py indexar` |
| `PDF sem texto selecionável` | O PDF é escaneado: rode OCR antes (ex.: `ocrmypdf manual.pdf manual_ocr.pdf`) |
| `nenhuma resposta aprovada` | Marque `"aprovada": true` no `resposta.json` |

## Referências

- Cortes, Vieira e Barone (2024). Perguntas e Respostas. Cap. 16 de *Processamento de Linguagem Natural*, 2ª ed.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Robertson e Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Cormack, Clarke e Büttcher (2009). Reciprocal Rank Fusion.
- Rajpurkar et al. (2016). SQuAD (EM e F1).
