"""Entrada e saída: JSONL/JSON, hashes, carimbos de tempo, run_id e log de execução.

Regra de confidencialidade: logs registram ids, contagens, tempos e hashes; nunca texto do manual.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
FORMATO_LOG = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ErroUsuario(Exception):
    """Erro de entrada ou de ambiente com mensagem que diz o que fazer.

    A CLI imprime a mensagem e encerra com ``codigo`` (2 para entradas inválidas).
    """

    codigo = 2


def pasta_sincronizada(caminho: str | Path) -> str | None:
    """Nome do serviço de nuvem que sincroniza ``caminho``, ou ``None`` se a pasta é só local.

    Detecta iCloud Drive (inclusive Mesa e Documentos sincronizados no macOS), pastas em
    ``~/Library/CloudStorage`` (Dropbox, OneDrive, Google Drive) e as pastas clássicas desses serviços. No
    iCloud, um componente terminado em ``.nosync`` exclui a pasta da sincronização.
    """
    p = Path(caminho).expanduser().resolve()
    casa = Path.home().resolve()
    icloud = casa / "Library" / "Mobile Documents"
    servico = None
    if p.is_relative_to(icloud):
        servico = "iCloud Drive"
    for pasta in ("Desktop", "Documents"):
        if (icloud / "com~apple~CloudDocs" / pasta).is_symlink() and p.is_relative_to(casa / pasta):
            servico = f"iCloud Drive (pasta {pasta} sincronizada)"
    if servico and any(parte.endswith(".nosync") for parte in p.parts):
        servico = None
    nuvens = casa / "Library" / "CloudStorage"
    if p.is_relative_to(nuvens) and p != nuvens:
        servico = p.relative_to(nuvens).parts[0]
    for nome in ("Dropbox", "OneDrive", "Google Drive"):
        if p.is_relative_to(casa / nome):
            servico = nome
    return servico


# ---------------------------------------------------------------------------
# Arquivos
# ---------------------------------------------------------------------------


def validar_arquivo(caminho: str | Path, extensoes: tuple[str, ...] = ()) -> Path:
    """Confere se o arquivo existe e tem uma das extensões esperadas.

    Args:
        caminho: caminho informado pelo usuário.
        extensoes: extensões aceitas (com ponto, minúsculas); vazio aceita qualquer uma.

    Returns:
        O caminho como ``Path``.

    Raises:
        ErroUsuario: se o arquivo não existe, não é arquivo ou tem extensão inesperada.
    """
    p = Path(caminho).expanduser()
    if not p.exists():
        raise ErroUsuario(f"Arquivo não encontrado: {p}")
    if not p.is_file():
        raise ErroUsuario(f"O caminho não é um arquivo: {p}")
    if extensoes and p.suffix.lower() not in extensoes:
        raise ErroUsuario(f"Extensão inválida para {p.name}: esperado {', '.join(extensoes)}.")
    return p


def ler_jsonl(caminho: str | Path) -> list[dict]:
    """Lê um arquivo JSONL (uma linha por registro), ignorando linhas vazias.

    Raises:
        ErroUsuario: se alguma linha não for um objeto JSON válido (a mensagem cita só o número da linha).
    """
    itens = []
    with open(caminho, encoding="utf-8") as f:
        for n, linha in enumerate(f, 1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                item = json.loads(linha)
            except json.JSONDecodeError as e:
                raise ErroUsuario(f"{caminho}, linha {n}: JSON inválido ({e.msg}).") from None
            if not isinstance(item, dict):
                raise ErroUsuario(f"{caminho}, linha {n}: esperado um objeto JSON.")
            itens.append(item)
    return itens


def escrever_jsonl(caminho: str | Path, itens: Iterable[dict]) -> None:
    """Grava os registros em JSONL (UTF-8), criando o diretório se preciso."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        for item in itens:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def anexar_jsonl(caminho: str | Path, item: dict) -> None:
    """Acrescenta um registro e força a escrita, para sobreviver a interrupções."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def ler_json(caminho: str | Path) -> Any:
    """Lê um arquivo JSON."""
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def escrever_json(caminho: str | Path, dados: Any) -> None:
    """Grava ``dados`` como JSON indentado (UTF-8)."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def sha256_arquivo(caminho: str | Path) -> str:
    """Hash SHA-256 do conteúdo do arquivo."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha256_texto(texto: str) -> str:
    """Hash SHA-256 de um texto (UTF-8)."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def lotes(itens: list, tamanho: int) -> Iterator[list]:
    """Divide ``itens`` em lotes consecutivos de até ``tamanho`` elementos."""
    for i in range(0, len(itens), tamanho):
        yield itens[i : i + tamanho]


# ---------------------------------------------------------------------------
# Tempo, execuções e versão
# ---------------------------------------------------------------------------


def agora_iso() -> str:
    """Instante atual em UTC no formato ``2026-10-06T14:30:00Z``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def carimbo() -> str:
    """Carimbo local usado em nomes de execução: ``2026-10-10T18-02-11``."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def novo_run_id(*partes: str) -> str:
    """Cria um ``run_id`` com timestamp, por exemplo ``2026-10-10T18-02-11_S3_ajustado``."""
    return "_".join([carimbo(), *[p for p in partes if p]])


def criar_run_dir(runs_dir: str | Path, run_id: str) -> Path:
    """Cria (se preciso) e devolve ``runs_dir/run_id``."""
    caminho = Path(runs_dir) / run_id
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def versao_codigo() -> str:
    """Hash curto do commit atual (``git rev-parse --short HEAD``) ou ``'sem-git'``."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return "sem-git"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "sem-git"


def anexar_log_arquivo(caminho: str | Path) -> logging.Handler:
    """Passa a gravar o log do pacote ``qa_manual`` também em ``caminho`` (por exemplo ``log.txt``).

    Returns:
        O handler criado, para ser removido com :func:`remover_log_arquivo` ao fim da execução.
    """
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(caminho, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMATO_LOG))
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("qa_manual")
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    return handler


def remover_log_arquivo(handler: logging.Handler) -> None:
    """Desliga e fecha um handler criado por :func:`anexar_log_arquivo`."""
    logging.getLogger("qa_manual").removeHandler(handler)
    handler.close()
