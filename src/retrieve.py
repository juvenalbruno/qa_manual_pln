"""Etapa 3: recuperação BM25, densa e híbrida (Reciprocal Rank Fusion)."""

from __future__ import annotations

import numpy as np

from .index import Indice, embeddings
from .ollama_client import OllamaClient
from .utils import tokenizar_bm25

MODOS = ("bm25", "denso", "hibrido")


class Recuperador:
    def __init__(self, indice: Indice, cliente: OllamaClient | None = None, rrf_k: int = 60):
        self.indice = indice
        self.cliente = cliente
        self.rrf_k = rrf_k
        self._cache_vetor: dict[str, np.ndarray] = {}

    def bm25(self, pergunta: str, k: int) -> list[tuple[str, float]]:
        toks = tokenizar_bm25(pergunta)
        n = len(self.indice.passagens)
        if not toks or n == 0:
            return []
        docs, scores = self.indice.bm25.retrieve([toks], k=min(k, n), show_progress=False)
        return [
            (self.indice.passagens[int(d)]["id"], float(s))
            for d, s in zip(docs[0], scores[0])
            if s > 0
        ]

    def vetor(self, pergunta: str) -> np.ndarray:
        if pergunta not in self._cache_vetor:
            if self.cliente is None:
                raise RuntimeError("Recuperação densa requer um cliente Ollama.")
            self._cache_vetor[pergunta] = embeddings(self.cliente, self.indice.modelo_embeddings, [pergunta])
        return self._cache_vetor[pergunta]

    def denso(self, pergunta: str, k: int) -> list[tuple[str, float]]:
        if self.indice.denso is None:
            raise RuntimeError("Índice denso não carregado.")
        k = min(k, self.indice.denso.ntotal)
        scores, pos = self.indice.denso.search(self.vetor(pergunta), k)
        return [(self.indice.ids_denso[int(p)], float(s)) for p, s in zip(pos[0], scores[0]) if p >= 0]

    def hibrido(self, pergunta: str, k: int, profundidade: int | None = None) -> list[tuple[str, float]]:
        profundidade = profundidade or max(50, 4 * k)
        listas = [self.bm25(pergunta, profundidade), self.denso(pergunta, profundidade)]
        return rrf(listas, self.rrf_k)[:k]

    def recuperar(self, pergunta: str, modo: str, k: int) -> list[tuple[str, float]]:
        if modo == "nenhum":
            return []
        if modo == "bm25":
            return self.bm25(pergunta, k)
        if modo == "denso":
            return self.denso(pergunta, k)
        if modo == "hibrido":
            return self.hibrido(pergunta, k)
        raise ValueError(f"Modo de recuperação desconhecido: {modo}")


def rrf(listas: list[list[tuple[str, float]]], constante: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (constante + posição), posição a partir de 1."""
    scores: dict[str, float] = {}
    for lista in listas:
        for posicao, (pid, _) in enumerate(lista, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (constante + posicao)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
