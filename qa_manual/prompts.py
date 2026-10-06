"""Templates de prompt em ``prompts/``: carga e preenchimento com ``str.format``.

Chaves literais nos arquivos (JSON de exemplo, formato da citação) são escritas como ``{{`` e ``}}``.
"""

from __future__ import annotations

import string
from functools import cache
from pathlib import Path

from .io_utils import RAIZ, ErroUsuario

DIR_PROMPTS = RAIZ / "prompts"

# Placeholders esperados em cada template (conferidos nos testes).
PLACEHOLDERS = {
    "leitor": {"trechos", "pergunta"},
    "leitor_s0": {"pergunta"},
    "gerador_perguntas": {"tipo", "texto"},
    "gerador_sem_resposta": {"texto"},
    "gerador_pares_treino": {"tipo", "tema_id", "pagina", "texto"},
}


@cache
def carregar(nome: str) -> str:
    """Lê ``prompts/<nome>.txt`` (o sufixo ``.txt`` é opcional).

    Raises:
        ErroUsuario: se o arquivo não existir.
    """
    arquivo = DIR_PROMPTS / (nome if nome.endswith(".txt") else f"{nome}.txt")
    if not arquivo.is_file():
        raise ErroUsuario(f"Prompt não encontrado: {arquivo}")
    return arquivo.read_text(encoding="utf-8")


def placeholders(template: str) -> set[str]:
    """Nomes dos campos ``{campo}`` do template (chaves escapadas não contam)."""
    return {campo for _, campo, _, _ in string.Formatter().parse(template) if campo}


def preencher(template: str, **campos: object) -> str:
    """Preenche o template com ``str.format``; valores com chaves não são reprocessados.

    Raises:
        ErroUsuario: se faltar algum campo do template.
    """
    try:
        return template.format(**campos)
    except KeyError as e:
        raise ErroUsuario(f"Campo {e} ausente ao preencher o prompt.") from None


def caminho(nome: str) -> Path:
    """Caminho do arquivo de um prompt."""
    return DIR_PROMPTS / f"{nome}.txt"
