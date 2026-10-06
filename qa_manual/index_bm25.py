"""Passo 4a: índice léxico BM25 (``bm25s``) em ``index/bm25/`` e ``index/bm25_ids.json``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s

from .chunking import Trecho
from .io_utils import ErroUsuario, escrever_json, ler_json
from .tokenize_pt import tokenizar_cfg

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)
_VAZIO = "_vazio_"  # token de um trecho sem tokens (o bm25s não aceita documento vazio)


class IndiceBM25:
    """Índice BM25 carregado em memória, com o mapeamento posição -> id do trecho."""

    def __init__(self, retriever: bm25s.BM25, ids: list[str], cfg: Config):
        self.retriever = retriever
        self.ids = ids
        self.cfg = cfg

    def buscar(self, pergunta: str, k: int) -> list[tuple[str, float]]:
        """Top-k trechos por BM25.

        Returns:
            ``[(id, score)]`` em ordem decrescente, só com score positivo (sem nenhum termo em comum, a
            lista fica vazia).
        """
        tokens = tokenizar_cfg(pergunta, self.cfg)
        if not tokens or not self.ids:
            return []
        docs, scores = self.retriever.retrieve([tokens], k=min(k, len(self.ids)), show_progress=False)
        return [(self.ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0], strict=True) if s > 0]


def _dir(cfg) -> Path:
    return Path(cfg.paths.index_dir)


def construir(trechos: list[Trecho], cfg: Config) -> IndiceBM25:
    """Indexa os trechos e grava ``index/bm25/`` e ``index/bm25_ids.json``."""
    corpus = [tokenizar_cfg(t.texto, cfg) or [_VAZIO] for t in trechos]
    retriever = bm25s.BM25(k1=cfg.bm25.k1, b=cfg.bm25.b)
    retriever.index(corpus, show_progress=False)
    destino = _dir(cfg)
    retriever.save(str(destino / "bm25"))
    ids = [t.id for t in trechos]
    escrever_json(destino / "bm25_ids.json", ids)
    log.info("BM25: %d trechos indexados (k1=%s, b=%s)", len(ids), cfg.bm25.k1, cfg.bm25.b)
    return IndiceBM25(retriever, ids, cfg)


def carregar(cfg: Config) -> IndiceBM25:
    """Carrega o índice gravado por :func:`construir`."""
    destino = _dir(cfg)
    if not (destino / "bm25").is_dir() or not (destino / "bm25_ids.json").is_file():
        raise ErroUsuario("índice não encontrado; rode `qa-manual indexar --manual <pdf>`")
    retriever = bm25s.BM25.load(str(destino / "bm25"))
    return IndiceBM25(retriever, ler_json(destino / "bm25_ids.json"), cfg)
