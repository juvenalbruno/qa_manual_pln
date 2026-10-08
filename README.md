# QA sobre o manual

Sistema de perguntas e respostas sobre um manual técnico do setor portuário, feito para a disciplina IC0024
(PGCOMP/UFBA). Ele lê o `manual.pdf`, busca os trechos mais relevantes para cada pergunta e pede a um modelo de
linguagem local (Ollama) que responda usando só esses trechos, citando seção e página.

Tudo roda na sua máquina: o manual não é enviado para nenhum serviço externo.

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `qa_manual.py` | O sistema de QA: responde perguntas sobre o manual e gera respostas para revisão |
| `treinar.py` | Ajuste fino (LoRA) do leitor com o manual inteiro; cria o modelo `qwen3-manual` |
| `avaliar.py` | Avaliação: EM, F1, BERTScore, Recall@5, perplexidade e outras; compara o leitor original com o ajustado |
| `requirements.txt` | Bibliotecas Python |
| `manual.pdf` | O manual (você coloca aqui; não vai para o git) |
| `perguntas.json` | As perguntas de avaliação (você escreve; não vai para o git) |

## Instalação (macOS, uma vez)

```bash
brew install python@3.11 ollama
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Não é preciso ligar o Ollama nem baixar modelos à mão: o programa liga o Ollama quando ele estiver desligado (e
desliga ao terminar) e baixa na primeira vez os modelos que faltarem (o leitor `qwen3:4b-instruct-2507-q4_K_M`,
`bge-m3` e `llama3.2:3b`; uns 6 GB).
O BERTimbau usado pelo BERTScore (~430 MB) também é baixado na primeira avaliação. São downloads de modelos: nenhum
texto do manual sai da máquina.

O treino e a perplexidade precisam de mais um passo, o llama.cpp (sem ele, o `avaliar.py` roda normalmente e deixa a
coluna `perplexidade` vazia):

```bash
brew install cmake
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=OFF
cmake --build ~/llama.cpp/build --target llama-perplexity llama-export-lora llama-quantize -j
```

Em cada terminal novo, ative o ambiente com `source .venv/bin/activate`. No Linux, instale o Ollama com
`curl -fsSL https://ollama.com/install.sh | sh`.

Não deixe a pasta do projeto na Mesa ou em Documentos se o iCloud estiver sincronizando essas pastas (nem em
Dropbox, OneDrive ou Google Drive): o manual e o índice iriam para a nuvem.

## Uso

```bash
python qa_manual.py              # 1. fazer perguntas
python qa_manual.py responder    # 2. perguntas.json -> resposta.json (para revisar)
python treinar.py                # 3. ajuste fino com o manual (uma vez)
python avaliar.py                # 4. métricas: leitor original x ajustado
```

### 1. Fazer perguntas

Coloque o `manual.pdf` ao lado do `qa_manual.py` e rode `python qa_manual.py`. Na primeira vez (e sempre que o
`manual.pdf` mudar) ele indexa o manual antes, o que leva alguns minutos.

```
Pergunta:
Qual o prazo para liberação da carga?
Resposta: A carga é liberada em até 48 horas após a atracação (seção 4.2, p. 17).

Pergunta:
```

Linha vazia ou `sair` encerra.

### 2. Montar o gabarito

**a) Escreva o `perguntas.json`** ao lado do `qa_manual.py` (de 50 a 100 perguntas; inclua algumas que o manual
não responde, para medir se o sistema sabe dizer que não sabe):

```json
[
  {"id": 1, "pergunta": "Qual o prazo para liberação da carga?"},
  {"id": 2, "pergunta": "Qual o salário do operador de guindaste?"}
]
```

**b) Rode `python qa_manual.py responder`.** Cria o `resposta.json` com uma resposta proposta para cada pergunta,
escrita pelo `llama3.2:3b` (que não é o leitor avaliado: assim o leitor não é comparado com as próprias respostas).
Ele não sobrescreve um `resposta.json` existente (use `--forcar` para gerar de novo).

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

### 3. Treinar

```bash
python treinar.py
```

Ajusta o leitor ao manual (ajuste fino LoRA, em Mac com Apple Silicon, via MLX). Usa o manual inteiro: cada trecho
do índice vira uma conversa curta ("O que diz o manual na seção X (p. N)?" → o texto do trecho), e o modelo treina
por 2 passadas sobre todos eles. No fim, o modelo ajustado é registrado no Ollama como `qwen3-manual`.

- Roda uma vez (de novo só se o manual mudar). Leva de minutos a algumas horas, conforme o tamanho do manual.
- Disco: o adaptador treinado tem poucos MB e é somado direto ao arquivo do leitor que o Ollama já tem; as camadas
  treinadas voltam a ser compactadas em 4 bits. O pico é de ~6 GB livres durante o treino e, no fim, o `qwen3-manual`
  ocupa ~2,5 GB (o mesmo tamanho do leitor original). Os arquivos intermediários são apagados.
- Baixa os pesos MLX do leitor (~2,3 GB) para treinar e apaga ao terminar. Nenhum texto do manual sai da máquina.
- Treino mais curto: `python treinar.py --epocas 1`.
- Para conversar com o modelo ajustado: `python qa_manual.py perguntar --modelo qwen3-manual`.

Os dados e o adaptador ficam em `modelos/` (não vai para o git).

### 4. Avaliar

```bash
python avaliar.py
```

Roda cada pergunta aprovada nos quatro modos de busca, **S0** sem busca (o modelo responde sozinho), **S1** BM25
(palavras), **S2** denso (embeddings) e **S3** híbrido (BM25 + denso fundidos por RRF), e mede a perplexidade do
leitor sobre o texto do manual. Se o `qwen3-manual` já existir, ele é avaliado junto com o leitor original, para
comparar com e sem treino. Os resultados ficam em `resultados/<data-hora>/`:

| Arquivo | Conteúdo |
|---|---|
| `metrics.csv` | A tabela de métricas: uma linha por modo de busca (e por modelo, se comparar mais de um) |
| `grafico.png` | Barras de EM, F1, BERTScore e Recall@5 para cada modo |
| `grafico_perplexidade.png` | Barra da perplexidade de cada modelo |
| `respostas.jsonl` | Cada resposta dada, com o gabarito, os trechos buscados e o BERTScore |
| `ppl_texto.txt` | Os 30 trechos do manual usados na perplexidade |

A mesma tabela do `metrics.csv` aparece no terminal ao final.

| Métrica | O que mede |
|---|---|
| `em` | Resposta igual ao gabarito (após normalizar) |
| `f1` | Sobreposição de palavras com o gabarito (0 a 1) |
| `bertscore_f1` | Semelhança de sentido com o gabarito pelo BERTimbau (0 a 1); aceita respostas certas com outras palavras |
| `recall_at_5` | O trecho com a resposta está entre os 5 buscados |
| `mrr_at_5` | 1 / posição desse trecho na busca |
| `abst_correta` | Nas perguntas sem resposta, quantas vezes o sistema disse "Não encontrado no manual" |
| `abst_indevida` | Nas perguntas com resposta, quantas vezes ele disse "Não encontrado" sem necessidade |
| `citacao_valida` | A seção e a página citadas batem com um trecho buscado |
| `latencia_mediana_s` | Tempo mediano por pergunta |
| `perplexidade` | Quanto o modelo "se surpreende" com o texto do manual (menor = mais familiar); uma por modelo |

Como o `qwen3-manual` treinou com o manual inteiro, a perplexidade dele mostra o quanto ele absorveu o texto, e não
se ele generaliza para textos novos. O ganho que interessa ao QA aparece nas outras métricas (EM, F1, BERTScore).

Opções, se precisar:

| Opção | Para quê |
|---|---|
| `--modos S0 S3 --limite 30` | Versão rápida: só dois modos e 30 perguntas |
| `--modelos modelo1 modelo2` | Escolhe os leitores (padrão: o leitor original e, se existir, `qwen3-manual`) |
| `--sem-bertscore`, `--sem-perplexidade` | Pula essas métricas |
| `--llama-perplexity caminho` | Se o llama.cpp não estiver em `~/llama.cpp` |

## Problemas comuns

| Mensagem | O que fazer |
|---|---|
| `Ollama não instalado` | `brew install ollama` |
| `manual não encontrado` | Coloque o `manual.pdf` ao lado do `qa_manual.py` |
| `PDF sem texto selecionável` | O PDF é escaneado: rode OCR antes (ex.: `ocrmypdf manual.pdf manual_ocr.pdf`) |
| `nenhuma resposta aprovada` | Marque `"aprovada": true` no `resposta.json` |
| `llama-perplexity não encontrado` | Compile o llama.cpp (instalação) ou use `--llama-perplexity` |
| `... não encontrado em ~/llama.cpp` | Baixe e compile o llama.cpp (instalação) |
| `mlx-lm não instalado` | O treino só roda em Mac com Apple Silicon |

## Referências

- Cortes, Vieira e Barone (2024). Perguntas e Respostas. Cap. 16 de *Processamento de Linguagem Natural*, 2ª ed.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Robertson e Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Cormack, Clarke e Büttcher (2009). Reciprocal Rank Fusion.
- Rajpurkar et al. (2016). SQuAD (EM e F1).
- Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
