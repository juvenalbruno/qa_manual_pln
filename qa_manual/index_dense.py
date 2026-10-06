"""Passo 4b: índice denso (bge-m3 via Ollama + FAISS ``IndexFlatIP``) em ``index/dense.faiss``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from tqdm import tqdm

from .chunking import Trecho
from .io_utils import ErroUsuario, escrever_json, ler_json, lotes
from .ollama_client import OllamaClient

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


def normalizar_l2(m: np.ndarray) -> np.ndarray:
    """Divide cada linha pela sua norma L2 (linhas nulas ficam como estão)."""
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return (m / normas).astype(np.float32)


def embed(textos: list[str], cfg: Config, client: OllamaClient, progresso: bool = False) -> np.ndarray:
    """Embeddings dos textos em lotes de ``denso.batch_size``.

    Returns:
        Matriz ``float32`` de forma ``(len(textos), dimensao)``, normalizada em L2 se configurado.

    Raises:
        ErroUsuario: se a dimensão devolvida pelo modelo diferir de ``denso.dimensao``.
    """
    vetores: list[list[float]] = []
    grupos = list(lotes(textos, cfg.denso.batch_size))
    for grupo in tqdm(grupos, desc="embeddings", unit="lote", disable=not progresso):
        vetores.extend(client.embed(cfg.modelos.embeddings, input=grupo))
    m = np.asarray(vetores, dtype=np.float32)
    if m.ndim != 2 or m.shape[1] != cfg.denso.dimensao:
        recebida = m.shape[1] if m.ndim == 2 else "?"
        raise ErroUsuario(
            f"O modelo {cfg.modelos.embeddings} devolveu vetores de dimensão {recebida}, "
            f"mas denso.dimensao = {cfg.denso.dimensao}."
        )
    return normalizar_l2(m) if cfg.denso.normalizar_l2 else m


class IndiceDenso:
    """Índice FAISS carregado em memória."""

    def __init__(self, index: faiss.Index, ids: list[str], cfg: Config, client: OllamaClient):
        if index.ntotal != len(ids):
            raise ErroUsuario("index/dense.faiss e index/dense_ids.json estão inconsistentes; rode `indexar`.")
        self.index = index
        self.ids = ids
        self.cfg = cfg
        self.client = client

    def vetor(self, pergunta: str) -> np.ndarray:
        """Embedding da pergunta (forma ``1 x d``)."""
        return embed([pergunta], self.cfg, self.client)

    def buscar_vetor(self, v: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top-k por produto interno (igual ao cosseno com vetores normalizados)."""
        k = min(k, self.index.ntotal)
        if k <= 0:
            return []
        scores, pos = self.index.search(v, k)
        return [(self.ids[int(p)], float(s)) for p, s in zip(pos[0], scores[0], strict=True) if p >= 0]

    def buscar(self, pergunta: str, k: int) -> list[tuple[str, float]]:
        """Top-k trechos para a pergunta: ``[(id, score)]``."""
        return self.buscar_vetor(self.vetor(pergunta), k)


def construir(trechos: list[Trecho], cfg: Config, client: OllamaClient) -> IndiceDenso:
    """Calcula os embeddings dos trechos e grava ``index/dense.faiss`` e ``index/dense_ids.json``."""
    matriz = embed([t.texto for t in trechos], cfg, client, progresso=True)
    index = faiss.IndexFlatIP(cfg.denso.dimensao)
    index.add(matriz)
    destino = Path(cfg.paths.index_dir)
    destino.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destino / "dense.faiss"))
    ids = [t.id for t in trechos]
    escrever_json(destino / "dense_ids.json", ids)
    log.info("denso: %d vetores de dimensão %d (%s)", len(ids), cfg.denso.dimensao, cfg.modelos.embeddings)
    return IndiceDenso(index, ids, cfg, client)


def carregar(cfg: Config, client: OllamaClient) -> IndiceDenso:
    """Carrega o índice gravado por :func:`construir`."""
    destino = Path(cfg.paths.index_dir)
    if not (destino / "dense.faiss").is_file() or not (destino / "dense_ids.json").is_file():
        raise ErroUsuario("índice não encontrado; rode `qa-manual indexar --manual <pdf>`")
    return IndiceDenso(
        faiss.read_index(str(destino / "dense.faiss")), ler_json(destino / "dense_ids.json"), cfg, client
    )
