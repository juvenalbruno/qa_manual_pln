"""Passos 8 e 9: recuperação BM25, densa e híbrida (Reciprocal Rank Fusion)."""

from __future__ import annotations

import time

from .index_bm25 import IndiceBM25
from .index_dense import IndiceDenso
from .io_utils import ErroUsuario

MODOS = ("nenhum", "bm25", "denso", "hibrido")


def rrf(listas: list[list[tuple[str, float]]], c: int, k: int) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    ``score(id) = soma, sobre as listas em que o id aparece, de 1 / (c + posicao_1based)``. Empates são
    desfeitos pelo id, para que o resultado seja determinístico.

    Args:
        listas: listas ``[(id, score)]`` já ordenadas; o score original é ignorado.
        c: constante do RRF (60 no artigo original).
        k: tamanho da lista devolvida.

    Returns:
        Top-k ``[(id, score_rrf)]`` em ordem decrescente.
    """
    scores: dict[str, float] = {}
    for lista in listas:
        for posicao, (tid, _) in enumerate(lista, start=1):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (c + posicao)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def recuperar(
    pergunta: str, modo: str, cfg, bm25: IndiceBM25 | None, denso: IndiceDenso | None
) -> tuple[list[dict], dict]:
    """Recupera os trechos para a pergunta.

    Args:
        pergunta: pergunta em português.
        modo: ``nenhum``, ``bm25``, ``denso`` ou ``hibrido``.
        cfg: configuração (usa ``k``, ``k_candidatos_por_lista`` e ``rrf_c``).
        bm25: índice léxico (exigido em ``bm25`` e ``hibrido``).
        denso: índice denso (exigido em ``denso`` e ``hibrido``).

    Returns:
        ``([{'id', 'score', 'rank'}], {'t_embedding_s', 't_busca_s'})``. Em ``nenhum``, lista vazia e tempos
        zero. No híbrido, cada lista traz ``k_candidatos_por_lista`` antes da fusão.
    """
    tempos = {"t_embedding_s": 0.0, "t_busca_s": 0.0}
    if modo == "nenhum":
        return [], tempos
    if modo not in MODOS:
        raise ErroUsuario(f"Modo de recuperação desconhecido: {modo}. Use: {', '.join(MODOS)}.")
    if modo in ("bm25", "hibrido") and bm25 is None:
        raise ErroUsuario(f"O modo {modo} exige o índice BM25.")
    if modo in ("denso", "hibrido") and denso is None:
        raise ErroUsuario(f"O modo {modo} exige o índice denso.")

    profundidade = cfg.k_candidatos_por_lista if modo == "hibrido" else cfg.k
    lista_bm25: list[tuple[str, float]] = []
    lista_densa: list[tuple[str, float]] = []
    if modo in ("bm25", "hibrido"):
        t0 = time.perf_counter()
        lista_bm25 = bm25.buscar(pergunta, profundidade)
        tempos["t_busca_s"] += time.perf_counter() - t0
    if modo in ("denso", "hibrido"):
        t0 = time.perf_counter()
        v = denso.vetor(pergunta)
        t1 = time.perf_counter()
        lista_densa = denso.buscar_vetor(v, profundidade)
        tempos["t_embedding_s"] += t1 - t0
        tempos["t_busca_s"] += time.perf_counter() - t1

    if modo == "bm25":
        final = lista_bm25[: cfg.k]
    elif modo == "denso":
        final = lista_densa[: cfg.k]
    else:
        t0 = time.perf_counter()
        final = rrf([lista_bm25, lista_densa], cfg.rrf_c, cfg.k)
        tempos["t_busca_s"] += time.perf_counter() - t0
    recuperadas = [{"id": tid, "score": round(s, 6), "rank": r} for r, (tid, s) in enumerate(final, start=1)]
    return recuperadas, {k: round(v, 4) for k, v in tempos.items()}
