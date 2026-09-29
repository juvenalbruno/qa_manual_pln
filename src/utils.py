"""Utilitários comuns: JSONL, normalização de texto, solicitação de arquivos e configurações."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROMPTS = RAIZ / "prompts"
DIR_CONFIGS = RAIZ / "configs"
DIR_RUNS = Path("runs")
DIR_INDEX = Path("index")

FRASE_ABSTENCAO = "Não encontrado no manual"


class ErroUsuario(Exception):
    """Erro causado por entrada inválida; a CLI imprime a mensagem e sai com código 2."""


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def ler_jsonl(caminho: str | Path) -> list[dict]:
    itens = []
    with open(caminho, encoding="utf-8") as f:
        for n, linha in enumerate(f, 1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                itens.append(json.loads(linha))
            except json.JSONDecodeError as e:
                raise ErroUsuario(f"{caminho}, linha {n}: JSON inválido ({e.msg}).") from e
    return itens


def escrever_jsonl(caminho: str | Path, itens: Iterable[dict]) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        for item in itens:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def anexar_jsonl(caminho: str | Path, item: dict) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def ler_json(caminho: str | Path) -> Any:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def escrever_json(caminho: str | Path, dados: Any) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

# Artigos e preposições frequentes removidos na normalização de EM/F1.
PALAVRAS_VAZIAS_METRICA = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "ao", "aos", "à", "às", "para", "pra", "por", "pelo", "pela", "pelos", "pelas",
    "com", "num", "numa", "dum", "duma",
}

# Stopwords usadas na tokenização do BM25 (sem acentos, pois o texto é desacentuado antes).
STOPWORDS_BM25 = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "ao", "aos", "a", "as", "para", "pra", "por", "pelo",
    "pela", "pelos", "pelas", "com", "sem", "e", "ou", "que", "se", "sao", "ser", "foi",
    "como", "qual", "quais", "quando", "onde", "quem", "e", "este", "esta", "isto", "esse",
    "essa", "isso", "aquele", "aquela", "seu", "sua", "seus", "suas", "ja", "nao", "mais",
    "muito", "tambem", "ha", "deve", "devem", "pode", "podem", "sobre", "entre", "ate",
    "apos", "num", "numa", "lhe", "the", "of", "and", "to", "in",
}

_PONTUACAO = re.compile(r"[%s–—‘’“”«»•·]" % re.escape(string.punctuation))


def sem_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar_resposta(texto: str) -> str:
    """Normalização para EM/F1: minúsculas, sem pontuação, sem artigos/preposições, espaços colapsados."""
    texto = texto.lower()
    texto = _PONTUACAO.sub(" ", texto)
    tokens = [t for t in texto.split() if t not in PALAVRAS_VAZIAS_METRICA]
    return " ".join(tokens)


def tokenizar_bm25(texto: str) -> list[str]:
    """Tokenização para o índice léxico: minúsculas, sem acentos, sem pontuação, sem stopwords."""
    texto = sem_acentos(texto.lower())
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return [t for t in texto.split() if t not in STOPWORDS_BM25 and (len(t) > 1 or t.isdigit())]


def contem_abstencao(resposta: str, frases: Iterable[str] = (FRASE_ABSTENCAO,)) -> bool:
    alvo = sem_acentos(resposta.lower())
    return any(sem_acentos(f.lower()) in alvo for f in frases)


def contar_palavras(texto: str) -> int:
    return len(texto.split())


def renderizar(template: str, **valores: Any) -> str:
    """Substitui apenas os marcadores {nome} conhecidos (o template pode conter chaves de JSON).

    A substituição é feita em uma única passada, para que um valor contendo "{x}" não seja reprocessado.
    """
    return re.sub(
        r"\{(\w+)\}",
        lambda m: str(valores[m.group(1)]) if m.group(1) in valores else m.group(0),
        template,
    )


def carregar_prompt(nome: str) -> str:
    caminho = DIR_PROMPTS / nome
    if not caminho.exists():
        raise ErroUsuario(f"Prompt não encontrado: {caminho}")
    return caminho.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Arquivos e configurações
# ---------------------------------------------------------------------------

def sha256_arquivo(caminho: str | Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def solicitar(valor: str | None, mensagem: str) -> str:
    """Devolve o valor passado por argumento ou, se omitido, pergunta ao usuário."""
    if valor:
        return valor
    if not sys.stdin.isatty():
        raise ErroUsuario(f"Argumento obrigatório ausente e sem terminal interativo: {mensagem}")
    try:
        resposta = input(f"{mensagem}: ").strip()
    except EOFError:
        resposta = ""
    if not resposta:
        raise ErroUsuario(f"Nenhum valor informado para: {mensagem}")
    return resposta


def solicitar_arquivo(valor: str | None, mensagem: str, extensoes: tuple[str, ...]) -> Path:
    """Solicita um arquivo de entrada e valida existência e extensão."""
    caminho = Path(solicitar(valor, mensagem)).expanduser()
    if not caminho.exists():
        raise ErroUsuario(f"Arquivo não encontrado: {caminho}")
    if not caminho.is_file():
        raise ErroUsuario(f"O caminho não é um arquivo: {caminho}")
    if caminho.suffix.lower() not in extensoes:
        raise ErroUsuario(
            f"Extensão inválida para {caminho.name}: esperado {', '.join(extensoes)}."
        )
    return caminho


def confirmar(mensagem: str, padrao: bool = False) -> bool:
    if not sys.stdin.isatty():
        return padrao
    sufixo = " [S/n] " if padrao else " [s/N] "
    try:
        resp = input(mensagem + sufixo).strip().lower()
    except EOFError:
        return padrao
    if not resp:
        return padrao
    return resp in {"s", "sim", "y", "yes"}


def carregar_yaml(caminho: str | Path) -> dict:
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def carregar_config(nome_ou_caminho: str) -> dict:
    """Carrega uma configuração pelo nome (S0..S3) ou pelo caminho de um YAML."""
    caminho = Path(nome_ou_caminho)
    if not caminho.suffix:
        caminho = DIR_CONFIGS / f"{nome_ou_caminho}.yaml"
    if not caminho.exists():
        disponiveis = sorted(p.stem for p in DIR_CONFIGS.glob("S*.yaml"))
        raise ErroUsuario(
            f"Configuração não encontrada: {nome_ou_caminho}. Disponíveis: {', '.join(disponiveis)}"
        )
    base = carregar_yaml(DIR_CONFIGS / "base.yaml") if (DIR_CONFIGS / "base.yaml").exists() else {}
    cfg = _ambiente({**base, **carregar_yaml(caminho)})
    cfg.setdefault("nome", caminho.stem)
    if cfg.get("modo") not in {"nenhum", "bm25", "denso", "hibrido"}:
        raise ErroUsuario(f"Configuração {cfg['nome']}: 'modo' deve ser nenhum, bm25, denso ou hibrido.")
    return cfg


def carregar_config_indexacao() -> dict:
    base = carregar_yaml(DIR_CONFIGS / "base.yaml") if (DIR_CONFIGS / "base.yaml").exists() else {}
    idx = carregar_yaml(DIR_CONFIGS / "indexacao.yaml") if (DIR_CONFIGS / "indexacao.yaml").exists() else {}
    return _ambiente({**base, **idx})


def _ambiente(cfg: dict) -> dict:
    """OLLAMA_HOST (mesma variável do Ollama) sobrescreve ollama_url; a URL continua restrita a hosts locais."""
    host = os.environ.get("OLLAMA_HOST")
    if host:
        cfg["ollama_url"] = host if host.startswith("http") else f"http://{host}"
    return cfg


def novo_run_dir(prefixo: str = "") -> tuple[str, Path]:
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if prefixo:
        run_id = f"{run_id}_{prefixo}"
    caminho = DIR_RUNS / run_id
    caminho.mkdir(parents=True, exist_ok=True)
    return run_id, caminho


def lotes(itens: list, tamanho: int) -> Iterator[list]:
    for i in range(0, len(itens), tamanho):
        yield itens[i : i + tamanho]


def tabela_markdown(cabecalho: list[str], linhas: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return "" if v is None else str(v)

    out = ["| " + " | ".join(cabecalho) + " |", "|" + "|".join("---" for _ in cabecalho) + "|"]
    for linha in linhas:
        out.append("| " + " | ".join(fmt(v) for v in linha) + " |")
    return "\n".join(out)
