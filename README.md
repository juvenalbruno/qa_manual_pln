# qa-manual

Sistema de perguntas e respostas em português sobre um manual técnico do setor portuário. Usa Retriever-Reader com
leitor generativo (RAG) e modelos abertos servidos localmente pelo Ollama. Trabalho da disciplina IC0024
(PGCOMP/UFBA).

O projeto roda em duas partes:

- **Local (CPU):** indexação do manual, perguntas, geração do conjunto de avaliação e avaliação. O manual nunca
  sai da máquina.
- **Google Colab (GPU T4):** só o ajuste fino do leitor, feito com um documento **público**. O resultado é um
  arquivo de modelo que volta para a máquina local.

> **Confidencialidade.** Não coloque o projeto em pasta sincronizada com a nuvem (Mesa ou Documentos com iCloud,
> Dropbox, OneDrive, Google Drive): o manual e os índices seriam enviados para fora da máquina, e o `indexar` se
> recusa a rodar nesse caso. Nunca envie o manual, nem nada de `data/`, `index/` ou `runs/`, para o Colab.

## Executar localmente

### O que precisa

| Item | Detalhe |
|---|---|
| Sistema | Testado em macOS (Apple Silicon). Linux deve funcionar; no Windows, use o WSL2 (não testado). |
| Hardware | CPU com 8 a 16 GB de RAM; GPU não é necessária. Cerca de 10 GB livres em disco (modelos e ambiente). |
| Python | 3.11 (`brew install python@3.11` no macOS; `sudo apt install python3.11 python3.11-venv` no Ubuntu). |
| Ollama | [ollama.com/download](https://ollama.com/download) no macOS, ou `curl -fsSL https://ollama.com/install.sh \| sh` no Linux. |
| Git | Para clonar o repositório. |
| llama.cpp (opcional) | Só para medir a perplexidade (passo 14). Exige `cmake` e um compilador C++. |

Modelos usados pelo Ollama: `qwen3:4b` (leitor), `bge-m3` (embeddings) e `llama3.2:3b` (gerador de perguntas). O
leitor ajustado, `qwen3-manual:4b`, vem do Colab (seção seguinte).

### Instalação

```bash
git clone https://github.com/SEU-USUARIO/qa-manual.git ~/projetos/qa-manual   # pasta local, fora da nuvem
cd ~/projetos/qa-manual

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -c "import nltk; nltk.download('stopwords')"

ollama serve                      # em outro terminal (no macOS, o aplicativo do Ollama já faz isso)
ollama pull qwen3:4b && ollama pull bge-m3 && ollama pull llama3.2:3b

qa-manual smoke                   # confere bibliotecas, Ollama, modelos e BERTScore
```

- Em Linux sem GPU, instale o PyTorch para CPU **antes** do `requirements.txt`:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- A primeira execução do BERTScore baixa o BERTimbau (~430 MB) do Hugging Face. É download de modelo, não envio
  de dados.
- Em cada novo terminal, ative o ambiente com `source .venv/bin/activate`.

Para a perplexidade (opcional), compile só o binário `llama-perplexity`:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
cmake -B build -DGGML_CUDA=OFF && cmake --build build --target llama-perplexity -j
# binário em ~/llama.cpp/build/bin/llama-perplexity
```

### Primeiro teste com o manual público

Antes de usar o manual da empresa, rode tudo com o manual público de exemplo (Porto de Salvador), que já vem com
151 perguntas revisadas:

```bash
qa-manual indexar --manual docs/exemplos/porto_salvador/manual_porto_salvador.pdf
qa-manual stats
qa-manual recuperar --pergunta "Qual a profundidade do Berço 202?" --modo hibrido
qa-manual perguntar --config-exp S3 --leitor base --pergunta "Qual a profundidade do Berço 202?"
qa-manual avaliar --gold docs/exemplos/porto_salvador/gold_dev.jsonl --configs S0 S3 --leitores base --limite 30
```

O `avaliar` mostra a tabela de métricas no terminal e grava tudo em `runs/<data>_avaliar/`. Ao passar para o
manual da empresa, rode o `indexar` de novo: ele substitui os trechos e o índice do exemplo.

### Experimento com o manual da empresa

```bash
mkdir -p data && cp /caminho/do/manual.pdf data/manual.pdf

# 1. Indexação (passos 1 a 4)
qa-manual indexar --manual data/manual.pdf
qa-manual stats                                    # confira quantos temas (seções) foram detectados

# 2. Conjunto de avaliação (passo 12)
qa-manual gerar-perguntas --n 150                  # candidatas geradas pelo llama3.2:3b
qa-manual exportar-revisao                         # gera data/gold_revisao.csv para revisão humana
#    Preencha a coluna "decisao" (aceitar, editar ou descartar) seguindo docs/protocolo_anotacao.md
qa-manual importar-revisao --csv data/gold_revisao.csv --dividir --reservar-ppl 30

# 3. Leitor ajustado: rode o notebook no Colab (seção seguinte) e registre o modelo
bash colab/criar_modelo_local.sh models/qwen3-manual-q4_k_m.gguf

# 4. Perplexidade (opcional, passo 14)
qa-manual perplexidade --binario ~/llama.cpp/build/bin/llama-perplexity \
    --gguf-base ollama --gguf-ajustado models/qwen3-manual-q4_k_m.gguf

# 5. Avaliação (passos 13 e 15)
qa-manual avaliar --gold data/gold_dev.jsonl --configs S3 --leitores base               # ajustes no dev
qa-manual avaliar --gold data/gold_test.jsonl --configs S0 S1 S2 S3 --leitores base ajustado
```

Resultados em `runs/<data>_avaliar/`: `metrics.csv` (uma linha por configuração e leitor), `pareado.csv` (ganho
do leitor ajustado), `respostas.jsonl` e dois gráficos PNG. Se a avaliação for interrompida, continue com
`--retomar <run_id>` (o nome da pasta em `runs/`).

Sem tempo para tudo, o **experimento mínimo** é: indexar, montar o gold e rodar
`qa-manual avaliar --gold data/gold_test.jsonl --configs S0 S3 --leitores base --limite 30`.

## Executar no Google Colab (ajuste fino)

O notebook [colab/finetune_qlora.ipynb](colab/finetune_qlora.ipynb) faz os passos 5 a 7. Ele gera pares de
treino a partir do documento público, ajusta o `Qwen3-4B` com QLoRA e exporta o leitor ajustado em GGUF.

### O que precisa

| Item | Detalhe |
|---|---|
| Conta Google | Com acesso ao [Google Colab](https://colab.research.google.com); a GPU T4 gratuita basta, sujeita a disponibilidade. |
| Documento público | O PDF de um regulamento portuário publicado por uma autoridade portuária. **Nunca o manual da empresa.** |
| Código do projeto | O repositório no GitHub (o notebook o clona) ou um `.zip` só com os arquivos versionados. |
| Tempo | Estimativa de 1 a 3 horas (ainda não medida). Geração dos pares e treino são as partes mais longas. |
| Espaço local | Cerca de 3 GB para baixar o GGUF ajustado. |

### Passo a passo

1. **Disponibilize o código.** Se o repositório estiver no GitHub, anote a URL e o branch. Se ele for privado,
   gere um `.zip` só com os arquivos versionados, que nunca inclui `data/`:
   `git archive --format=zip -o qa-manual.zip HEAD`. No Colab, envie o `.zip` pelo painel **Arquivos** e rode
   `!unzip -q qa-manual.zip -d qa-manual` numa célula antes da instalação; a clonagem é pulada.
2. **Abra o notebook.** No Colab: **Arquivo → Fazer upload de notebook** e escolha `colab/finetune_qlora.ipynb`
   (ou abra direto pela aba GitHub).
3. **Ative a GPU.** **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4**.
4. **Ajuste os parâmetros** na primeira célula de código: `REPO_URL` e `BRANCH` (o branch que tem este código).
5. **Confirme o documento.** Na seção 3 do notebook, mude `DOCUMENTO_E_PUBLICO = False` para `True` só depois de
   conferir que o PDF é público.
6. **Execute.** **Ambiente de execução → Executar tudo**. Quando aparecer o botão de upload, escolha o PDF público.
7. **Acompanhe** as saídas: o passo 5 informa quantos pares foram gerados e a taxa de JSON válido; o treino mostra
   a perda a cada 10 passos; a checagem de sanidade marca `OK` ou `!!` em 10 respostas de validação.
8. **Baixe os resultados.** A célula de download, na seção 7 do notebook, baixa `qwen3-manual-q4_k_m.gguf`,
   `treino_log.csv` e `metricas_validacao.json`. O navegador pode pedir permissão para vários downloads.

Se a sessão do Colab cair, os arquivos dela se perdem: rode o notebook de novo desde o início.

### De volta à máquina local

```bash
mkdir -p models && mv ~/Downloads/qwen3-manual-q4_k_m.gguf models/
mv ~/Downloads/treino_log.csv ~/Downloads/metricas_validacao.json colab/    # para o relatório
bash colab/criar_modelo_local.sh models/qwen3-manual-q4_k_m.gguf          # cria qwen3-manual:4b no Ollama
qa-manual smoke                                                           # agora sem aviso sobre o leitor ajustado
```

O script gera o `Modelfile` a partir do `qwen3:4b`, trocando só o arquivo do modelo, e registra
`qwen3-manual:4b` no Ollama. Depois disso, o leitor `ajustado` funciona em `perguntar` e `avaliar`.

## Testes

```bash
pytest --cov      # sem Ollama e sem o manual; usa um PDF sintético e um cliente Ollama falso
```

## Documentação

- [docs/README.md](docs/README.md): referência completa (pipeline, todos os comandos e saídas, configuração,
  formatos de arquivo, métricas, decisões de implementação e referências bibliográficas).
- [docs/protocolo_anotacao.md](docs/protocolo_anotacao.md): como revisar as perguntas no CSV.
- [docs/relatorio.md](docs/relatorio.md) e [docs/slides.md](docs/slides.md): modelos para a entrega.
- [docs/exemplos/](docs/exemplos/): manual público de teste, gold, casos de erro e perguntas de exemplo.
