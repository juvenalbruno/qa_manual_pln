"""Gera ``colab/finetune_qlora.ipynb``: o notebook do ajuste fino (passos 5 a 7), autossuficiente.

O código do projeto que o Colab precisa (módulos de ``qa_manual``, ``configs/base.yaml`` e os prompts) vai
embutido no notebook em células ``%%writefile``; o Colab não baixa nada do repositório. Rode de novo sempre que
mudar um dos arquivos de ``EMBUTIDOS`` (o teste ``tests/test_notebook_colab.py`` falha se o notebook divergir):

    python colab/gerar_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / "colab" / "finetune_qlora.ipynb"
PASTA_COLAB = "/content/qa_manual_colab"

# Arquivos do projeto copiados para o Colab: só o que os passos 1 a 3 e 5 usam (fecho de importações).
EMBUTIDOS = [
    "qa_manual/__init__.py",
    "qa_manual/io_utils.py",
    "qa_manual/config.py",
    "qa_manual/ollama_client.py",
    "qa_manual/ingest.py",
    "qa_manual/chunking.py",
    "qa_manual/tokenize_pt.py",
    "qa_manual/prompts.py",
    "qa_manual/generate.py",
    "qa_manual/verify.py",
    "qa_manual/gold.py",
    "configs/base.yaml",
    "prompts/leitor.txt",
    "prompts/leitor_s0.txt",
    "prompts/gerador_pares_treino.txt",
    "prompts/gerador_sem_resposta.txt",
]


def arquivos_embutidos() -> dict[str, str]:
    """Conteúdo atual de cada arquivo de ``EMBUTIDOS``."""
    return {caminho: (RAIZ / caminho).read_text(encoding="utf-8") for caminho in EMBUTIDOS}


def _celulas() -> list[tuple[str, str]]:
    celulas: list[tuple[str, str]] = []

    def md(texto: str) -> None:
        celulas.append(("markdown", texto.strip("\n")))

    def code(texto: str) -> None:
        celulas.append(("code", texto.strip("\n")))

    md(r"""
# Ajuste fino do leitor (QLoRA) — passos 5 a 7

IC0024 (PGCOMP/UFBA). Este notebook gera pares de treino a partir de um **documento público** (um regulamento
portuário publicado por uma autoridade portuária), ajusta o `Qwen3-4B` com QLoRA (Unsloth) e exporta o leitor
ajustado em GGUF `q4_k_m` para ser registrado no Ollama local como `qwen3-manual:4b`.

**Autossuficiente.** O código do projeto que este notebook usa vem embutido na seção 3; nada é baixado do
repositório. Da internet vêm só pacotes Python, o Ollama, o modelo `llama3.2:3b` e o `Qwen3-4B`.

> **Confidencialidade.** Este notebook roda na nuvem do Google. **Nunca** envie para cá o manual da empresa
> (`data/manual.pdf`), nem `data/trechos.jsonl`, nem qualquer arquivo derivado dele. Só o documento público entra
> aqui. Do Colab voltam apenas o GGUF, `treino_log.csv` e `metricas_validacao.json`.

**Antes de começar:** *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4*. Depois,
*Ambiente de execução → Executar tudo*; na seção 4, mude `DOCUMENTO_E_PUBLICO` para `True` e envie o PDF.

| Passo | O que faz | Saída |
|---|---|---|
| 1–3 | ingestão e trechos do documento público, com o mesmo código do projeto | `trechos_publico.jsonl` |
| 5 | `llama3.2:3b` gera pergunta + resposta; prompt do leitor com 3 a 5 trechos | `treino.jsonl`, `validacao.jsonl` |
| 6 | QLoRA (Unsloth) sobre `unsloth/Qwen3-4B` em 4 bits, perda só na resposta | `adapter/`, `treino_log.csv` |
| 7 | merge + GGUF `q4_k_m` | `qwen3-manual-q4_k_m.gguf` |
""")

    md("## 1. Parâmetros")
    code(r"""
N_MAX_TRECHOS = 1500  # trechos do documento público usados para gerar exemplos
FRAC_SEM_RESPOSTA = 0.20  # exemplos cujo alvo é exatamente "Não encontrado no manual."
FRAC_VALIDACAO = 0.05
MAX_TOKENS_CONTEXTO = 1400  # estimativa (1,3 token/palavra) dos trechos no prompt, para caber em 2048 tokens
SEED = 42
""")

    md("## 2. Instalação (Unsloth, Ollama e dependências do código do projeto)")
    code(r"""
%%capture
!pip install -q unsloth
!pip install -q trl datasets
!pip install -q pymupdf bm25s nltk ollama pyyaml tqdm
!apt-get -qq install -y zstd pciutils > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
""")
    code(r"""
import subprocess
import time
import urllib.request

servidor_ollama = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError("o Ollama não subiu no Colab")
!ollama pull llama3.2:3b
""")

    md(f"""
## 3. Código do projeto (embutido)

As células abaixo gravam em `{PASTA_COLAB}/` os arquivos do projeto que o Colab usa: os módulos de `qa_manual`
dos passos 1 a 3 e 5, o `configs/base.yaml` e os prompts. Elas foram geradas por `colab/gerar_notebook.py` a
partir do repositório; não edite aqui, edite o código local e gere o notebook de novo.
""")
    code(f"""
import os
import sys

PASTA_CODIGO = "{PASTA_COLAB}"
for sub in ("qa_manual", "configs", "prompts"):
    os.makedirs(os.path.join(PASTA_CODIGO, sub), exist_ok=True)
""")
    for caminho, conteudo in arquivos_embutidos().items():
        celulas.append(("code", f"%%writefile {PASTA_COLAB}/{caminho}\n{conteudo}"))
    code(r"""
sys.path.insert(0, PASTA_CODIGO)
import nltk

import qa_manual

assert qa_manual.__file__.startswith(PASTA_CODIGO), qa_manual.__file__
nltk.download("stopwords", quiet=True)
print("qa_manual", qa_manual.__version__, "carregado de", PASTA_CODIGO)
""")

    md(r"""
## 4. Documento público → trechos (passos 1 a 3)

Envie o PDF do regulamento portuário público. O mesmo código e os mesmos parâmetros de `configs/base.yaml` do
projeto são usados, para que os trechos tenham o formato que o leitor verá em produção.
""")
    code(r"""
from pathlib import Path

from google.colab import files

DOCUMENTO_E_PUBLICO = False  # mude para True só depois de confirmar que o PDF é um documento público
assert DOCUMENTO_E_PUBLICO, "Confirme que o PDF é público (nunca o manual da empresa) e rode a célula de novo."

enviado = files.upload()
assert len(enviado) == 1, "envie exatamente um PDF"
PDF_PUBLICO = Path(next(iter(enviado)))
proibidos = {"manual.pdf", "trechos.jsonl", "trechos_stats.json"}
assert PDF_PUBLICO.name.lower() not in proibidos and "data/" not in str(PDF_PUBLICO), (
    "Este nome é o do manual confidencial; o Colab só pode receber o documento público."
)
assert PDF_PUBLICO.suffix.lower() == ".pdf", "o arquivo precisa ser um PDF"
print(PDF_PUBLICO, PDF_PUBLICO.stat().st_size, "bytes")
""")
    code(r"""
from qa_manual import chunking, config, ingest
from qa_manual.io_utils import escrever_jsonl

cfg = config.carregar()  # configs/base.yaml embutido na seção 3
info = {}
paragrafos = ingest.construir_paragrafos(PDF_PUBLICO, cfg, info)
trechos = chunking.agrupar_trechos(paragrafos, cfg)
stats = chunking.estatisticas(trechos, cfg, info)
escrever_jsonl("trechos_publico.jsonl", [t.como_dict() for t in trechos])
print({k: stats[k] for k in ("n_paginas", "n_titulos", "n_trechos", "n_temas", "tokens")})
""")

    md(r"""
## 5. Passo 5 — pares de treino

Para cada trecho (até `N_MAX_TRECHOS`), o `llama3.2:3b` gera uma pergunta e a resposta com
`prompts/gerador_pares_treino.txt`. A resposta sempre termina com a fonte do próprio trecho, `(seção X, p. N)`,
preenchida pelo código. Em 20% dos trechos a pergunta é gerada com `prompts/gerador_sem_resposta.txt` e o alvo é
exatamente `Não encontrado no manual.`. O prompt de cada exemplo é montado por `qa_manual.generate.montar_prompt`,
o mesmo usado na inferência, com o trecho-evidência e 2 a 4 distratores recuperados por BM25 (ou sorteados), em
ordem aleatória.
""")
    code(r"""
import json
import random

import bm25s
from tqdm.auto import tqdm

from qa_manual import gold, prompts
from qa_manual.generate import montar_prompt, rotulo_secao
from qa_manual.ollama_client import OllamaReal
from qa_manual.tokenize_pt import tokenizar_cfg
from qa_manual.verify import remover_citacoes

rng = random.Random(SEED)
client = OllamaReal("http://localhost:11434", timeout_s=300)
GERADOR = cfg.modelos.gerador
FRASE = cfg.abstencao.frase + "."

bm25 = bm25s.BM25(k1=cfg.bm25.k1, b=cfg.bm25.b)
bm25.index([tokenizar_cfg(t.texto, cfg) or ["_vazio_"] for t in trechos], show_progress=False)


def distratores(pergunta, evidencia, n):
    toks = tokenizar_cfg(pergunta, cfg)
    candidatos = []
    if toks:
        docs, _ = bm25.retrieve([toks], k=min(len(trechos), n + 3), show_progress=False)
        candidatos = [trechos[int(i)] for i in docs[0] if trechos[int(i)].id != evidencia.id]
    resto = [t for t in trechos if t.id != evidencia.id and t not in candidatos]
    rng.shuffle(resto)
    escolhidos = []
    usados = evidencia.n_tokens
    for t in candidatos + resto:
        if len(escolhidos) == n:
            break
        if usados + t.n_tokens <= MAX_TOKENS_CONTEXTO:
            escolhidos.append(t)
            usados += t.n_tokens
    return escolhidos


elegiveis = [t for t in trechos if t.n_tokens >= 40]
amostra = rng.sample(elegiveis, min(N_MAX_TRECHOS, len(elegiveis)))
n_sem = round(len(amostra) * FRAC_SEM_RESPOSTA)
contagem, exemplos = {}, []
for i, t in enumerate(tqdm(amostra, desc="pares")):
    citacao = f"(seção {rotulo_secao(t)}, p. {t.pagina_inicio})"
    if i < n_sem:
        p = prompts.preencher(prompts.carregar("gerador_sem_resposta"), texto=t.texto)
        dados = gold.gerar_json(client, GERADOR, p, ("pergunta",), cfg, contagem)
        alvo = FRASE
    else:
        tipo = "procedimental" if i % 5 in (1, 3) else "factual"
        p = prompts.preencher(
            prompts.carregar("gerador_pares_treino"),
            tipo=tipo,
            tema_id=rotulo_secao(t),
            pagina=t.pagina_inicio,
            texto=t.texto,
        )
        dados = gold.gerar_json(client, GERADOR, p, ("pergunta", "resposta"), cfg, contagem)
        if dados:
            alvo = f"{remover_citacoes(dados['resposta']).rstrip(' .')} {citacao}"
    if not dados:
        continue
    contexto = [t] + distratores(dados["pergunta"], t, rng.randint(2, 4))
    rng.shuffle(contexto)
    exemplos.append({"prompt": montar_prompt(dados["pergunta"], contexto), "resposta": alvo})

rng.shuffle(exemplos)
n_val = max(1, round(len(exemplos) * FRAC_VALIDACAO))
escrever_jsonl("validacao.jsonl", exemplos[:n_val])
escrever_jsonl("treino.jsonl", exemplos[n_val:])
print(
    f"{len(exemplos)} exemplos ({n_val} de validação); JSON válido em "
    f"{contagem.get('validas', 0)}/{contagem.get('tentativas', 0)} tentativas"
)
print(exemplos[0]["prompt"][-400:], "\n→", exemplos[0]["resposta"])
""")
    code(r"""
# Libera a GPU: o Ollama não é mais necessário.
servidor_ollama.terminate()
!pkill -f "ollama serve" || true
""")

    md(r"""
## 6. Passo 6 — QLoRA com Unsloth

`unsloth/Qwen3-4B` em 4 bits, LoRA `r=16`, `alpha=16`, `dropout=0` nas projeções de atenção e MLP. Chat template
do Qwen3 com `enable_thinking=False`; a perda é calculada só nos tokens da resposta.
""")
    code(r"""
from unsloth import FastLanguageModel, is_bf16_supported  # importar antes de transformers/trl

MAX_SEQ = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B",
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
""")
    code(r"""
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "treino.jsonl", "validation": "validacao.jsonl"})


def formatar(ex):
    msgs = [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["resposta"]}]
    texto = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    return {"text": texto, "n_tokens": len(tokenizer(texto).input_ids)}


ds = ds.map(formatar, remove_columns=["prompt", "resposta"])
antes = {k: len(v) for k, v in ds.items()}
ds = ds.filter(lambda ex: ex["n_tokens"] <= MAX_SEQ).remove_columns("n_tokens")  # truncado perderia a resposta
print("exemplos (antes → depois do filtro de tamanho):", antes, {k: len(v) for k, v in ds.items()})

# Marcadores do turno, extraídos do próprio template (inclui o bloco <think> vazio do Qwen3).
amostra = tokenizer.apply_chat_template(
    [{"role": "user", "content": "PERGUNTA"}, {"role": "assistant", "content": "RESPOSTA"}],
    tokenize=False,
    enable_thinking=False,
)
PARTE_USUARIO = "<|im_start|>user\n"
PARTE_RESPOSTA = amostra[amostra.rindex("<|im_start|>assistant") : amostra.rindex("RESPOSTA")]
print(repr(PARTE_RESPOSTA))
""")
    code(r"""
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not is_bf16_supported(),  # T4: fp16=True, bf16=False
        bf16=is_bf16_supported(),
        seed=SEED,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        output_dir="saidas",
        report_to="none",
    ),
)
trainer = train_on_responses_only(trainer, instruction_part=PARTE_USUARIO, response_part=PARTE_RESPOSTA)

# Conferência da máscara: só a resposta deve aparecer nos rótulos.
rotulos = [t for t in trainer.train_dataset[0]["labels"] if t != -100]
print("rótulos do 1º exemplo:", repr(tokenizer.decode(rotulos)))
""")
    code(r"""
import pandas as pd

resultado = trainer.train()
log = pd.DataFrame(trainer.state.log_history)
colunas = [c for c in ("step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm") if c in log.columns]
log[colunas].to_csv("treino_log.csv", index=False)
print(log[colunas].dropna(how="all", subset=[c for c in ("loss", "eval_loss") if c in colunas]).tail(10))
model.save_pretrained("adapter")
tokenizer.save_pretrained("adapter")
""")

    md(r"""
## 7. Checagem de sanidade

Dez exemplos da validação respondidos pelo modelo ajustado (via `transformers`): a resposta deve trazer a fonte
no formato `(seção X, p. N)` e, nos exemplos sem resposta, usar a frase de abstenção.
""")
    code(r"""
from qa_manual.verify import detectar_abstencao, extrair_citacoes

FastLanguageModel.for_inference(model)
validacao = [json.loads(l) for l in open("validacao.jsonl", encoding="utf-8")][:10]
checagens = []
for ex in validacao:
    entrada = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer(entrada, return_tensors="pt").to("cuda")
    saida = model.generate(**ids, max_new_tokens=160, do_sample=False)
    resposta = tokenizer.decode(saida[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    sem_resposta = ex["resposta"] == FRASE
    ok = detectar_abstencao(resposta, cfg.abstencao.frase) if sem_resposta else bool(extrair_citacoes(resposta))
    checagens.append({"sem_resposta": sem_resposta, "formato_ok": ok, "resposta": resposta, "alvo": ex["resposta"]})
    print(("OK " if ok else "!! ") + resposta[:160])

metricas = {
    "eval_loss_final": next((h["eval_loss"] for h in reversed(trainer.state.log_history) if "eval_loss" in h), None),
    "train_loss": resultado.training_loss,
    "formato_ok": sum(c["formato_ok"] for c in checagens) / len(checagens),
    "n_treino": len(ds["train"]),
    "n_validacao": len(ds["validation"]),
}
json.dump(metricas, open("metricas_validacao.json", "w"), ensure_ascii=False, indent=2)
metricas
""")

    md(r"""
## 8. Passo 7 — exportação em GGUF `q4_k_m`

O Unsloth faz o merge do adaptador e a conversão com o llama.cpp (pode levar alguns minutos).
""")
    code(r"""
import glob
import shutil

model.save_pretrained_gguf("qwen3-manual", tokenizer, quantization_method="q4_k_m")
gerados = [g for g in glob.glob("**/*.gguf", recursive=True) if "q4_k_m" in g.lower()]
assert gerados, "nenhum GGUF q4_k_m encontrado"
shutil.copy(gerados[0], "qwen3-manual-q4_k_m.gguf")
!ls -lh qwen3-manual-q4_k_m.gguf
""")
    code(r"""
for nome in ("qwen3-manual-q4_k_m.gguf", "treino_log.csv", "metricas_validacao.json"):
    files.download(nome)
""")

    md(r"""
## 9. Na máquina local

```bash
mkdir -p models && mv ~/Downloads/qwen3-manual-q4_k_m.gguf models/
mv ~/Downloads/treino_log.csv ~/Downloads/metricas_validacao.json colab/
bash colab/criar_modelo_local.sh models/qwen3-manual-q4_k_m.gguf   # Modelfile = o do qwen3:4b com outro FROM
qa-manual smoke --sem-bertscore
```

O script gera o `Modelfile` a partir de `ollama show qwen3:4b --modelfile`, trocando só a linha `FROM`, cria
`qwen3-manual:4b` e faz um teste com `ollama run`.
""")
    return celulas


def construir() -> dict:
    """Notebook completo (formato nbformat 4) como dicionário."""
    celulas = []
    for tipo, fonte in _celulas():
        c = {"cell_type": tipo, "metadata": {}, "source": fonte.splitlines(keepends=True)}
        if tipo == "code":
            c.update(execution_count=None, outputs=[])
        celulas.append(c)
    return {
        "cells": celulas,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def serializar(nb: dict) -> str:
    """Texto do ``.ipynb`` gravado em disco."""
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    NOTEBOOK.write_text(serializar(construir()), encoding="utf-8")
    print(f"{NOTEBOOK.relative_to(RAIZ)}: {len(construir()['cells'])} células, {len(EMBUTIDOS)} arquivos embutidos")
