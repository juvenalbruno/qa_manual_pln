"""Tokenização em português para o BM25 (a mesma função é usada na indexação e na consulta)."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

from .io_utils import ErroUsuario

if TYPE_CHECKING:
    from .config import Config

_NAO_ALFANUMERICO = re.compile(r"[\W_]+")


def remover_acentos(texto: str) -> str:
    """Remove diacríticos (decomposição NFKD sem marcas combinantes)."""
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


@lru_cache(maxsize=2)
def stopwords_pt(manter_acentos: bool = True) -> frozenset[str]:
    """Stop-words do NLTK para português, carregadas uma vez por processo.

    Raises:
        ErroUsuario: se o corpus ``stopwords`` do NLTK não estiver instalado.
    """
    try:
        from nltk.corpus import stopwords

        palavras = stopwords.words("portuguese")
    except LookupError:
        raise ErroUsuario(
            "Lista de stop-words do NLTK ausente; rode: python -c \"import nltk; nltk.download('stopwords')\""
        ) from None
    palavras = [unicodedata.normalize("NFC", p.lower()) for p in palavras]
    if not manter_acentos:
        palavras = [remover_acentos(p) for p in palavras]
    return frozenset(palavras)


def tokenizar(texto: str, remover_stopwords: bool = True, manter_acentos: bool = True) -> list[str]:
    """Tokeniza para o índice léxico.

    Passos: minúsculas; remoção de diacríticos se ``manter_acentos=False``; tudo que não for letra ou dígito
    vira espaço; divisão em espaços; remoção de tokens de 1 caractere; remoção das stop-words do NLTK.

    Args:
        texto: texto de entrada.
        remover_stopwords: remove stop-words do português.
        manter_acentos: preserva acentos (``liberação`` e ``liberacao`` são tokens distintos).

    Returns:
        Lista de tokens.
    """
    t = unicodedata.normalize("NFC", texto.lower())
    if not manter_acentos:
        t = remover_acentos(t)
    tokens = [tok for tok in _NAO_ALFANUMERICO.sub(" ", t).split() if len(tok) > 1]
    if remover_stopwords:
        sw = stopwords_pt(manter_acentos)
        tokens = [tok for tok in tokens if tok not in sw]
    return tokens


def tokenizar_cfg(texto: str, cfg: Config) -> list[str]:
    """:func:`tokenizar` com as opções de ``cfg.bm25``."""
    return tokenizar(texto, cfg.bm25.remover_stopwords, cfg.bm25.manter_acentos)
