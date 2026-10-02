"""Orquestração dos passos 8 a 11 para uma pergunta: recuperar, ler, verificar e montar o registro 5.6."""

from __future__ import annotations

import logging
import time

from .generate import responder
from .indexacao import Indices
from .ollama_client import OllamaClient
from .retrieve import recuperar
from .verify import detectar_abstencao, extrair_citacoes, validar_fonte

log = logging.getLogger(__name__)


def perguntar(
    pergunta: str,
    cfg,
    leitor: str,
    indices: Indices | None,
    client: OllamaClient,
    q_id: str | None = None,
    tipo: str | None = None,
    run_id: str = "",
) -> dict:
    """Responde uma pergunta com a configuração ``cfg`` e o leitor indicado.

    Se ``abstencao.limiar_score`` estiver definido e o melhor score recuperado for menor que ele (ou nada for
    recuperado), o sistema se abstém sem chamar o LLM.

    Args:
        pergunta: pergunta em português.
        cfg: configuração experimental (``cfg.nome`` e ``cfg.retriever``).
        leitor: ``base`` ou ``ajustado``.
        indices: trechos e índices (pode ser ``None`` em S0).
        client: cliente Ollama.
        q_id: id da pergunta no gold, se houver.
        tipo: tipo da pergunta no gold, se houver.
        run_id: identificador da execução.

    Returns:
        Registro no contrato de ``runs/<run_id>/respostas.jsonl`` (seção 5.6).
    """
    inicio = time.perf_counter()
    modo = cfg.retriever
    bm25 = indices.bm25 if indices else None
    denso = indices.denso if indices else None
    recuperadas, tempos = recuperar(pergunta, modo, cfg, bm25, denso)
    trechos = [indices.por_id[r["id"]] for r in recuperadas] if recuperadas else []

    limiar = cfg.abstencao.limiar_score
    if modo != "nenhum" and limiar is not None and (not recuperadas or recuperadas[0]["score"] < limiar):
        resposta = f"{cfg.abstencao.frase}."
        leitura = {"t_leitura_s": 0.0, "n_tokens_prompt": 0, "n_tokens_resposta": 0}
        log.info("q=%s abstenção por limiar (melhor score abaixo de %s)", q_id, limiar)
    else:
        resposta, leitura = responder(pergunta, trechos, leitor, cfg, client)

    absteve = detectar_abstencao(resposta, cfg.abstencao.frase)
    citacoes = extrair_citacoes(resposta)
    registro = {
        "run_id": run_id,
        "config": cfg.nome,
        "retriever": modo,
        "leitor": leitor,
        "modelo": cfg.modelo_leitor(leitor),
        "q_id": q_id,
        "pergunta": pergunta,
        "tipo": tipo,
        "recuperadas": recuperadas,
        "resposta": resposta,
        "absteve": absteve,
        "citou_fonte": bool(citacoes),
        "fonte_valida": validar_fonte(citacoes, trechos),
        "citacoes": citacoes,
        "latencia_s": round(time.perf_counter() - inicio, 4),
        **tempos,
        **leitura,
    }
    log.info(
        "q=%s config=%s leitor=%s recuperadas=%s absteve=%s fonte_valida=%s latencia=%.2fs",
        q_id,
        cfg.nome,
        leitor,
        [r["id"] for r in recuperadas],
        absteve,
        registro["fonte_valida"],
        registro["latencia_s"],
    )
    return registro
