"""Passo 10: montagem do prompt do leitor e geração da resposta pelo Ollama."""

from __future__ import annotations

import logging
import re
import time

from . import prompts
from .chunking import Trecho
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)

_THINK = re.compile(r"<think>.*?(?:</think>|$)\s*", re.DOTALL)


def rotulo_secao(t: Trecho) -> str:
    """Seção exibida ao leitor: a numeração do tema ou, sem numeração, o próprio título."""
    return t.tema_id or t.tema


def formatar_trechos(trechos: list[Trecho]) -> str:
    """Renderiza cada trecho como ``[n] (seção {tema_id}, p. {pagina_inicio}) {texto}``."""
    return "\n\n".join(
        f"[{n}] (seção {rotulo_secao(t)}, p. {t.pagina_inicio}) {t.texto}" for n, t in enumerate(trechos, start=1)
    )


def montar_prompt(pergunta: str, trechos: list[Trecho]) -> str:
    """Prompt do leitor: ``leitor.txt`` com trechos ou ``leitor_s0.txt`` quando não há trechos (S0)."""
    if not trechos:
        return prompts.preencher(prompts.carregar("leitor_s0"), pergunta=pergunta.strip())
    return prompts.preencher(prompts.carregar("leitor"), trechos=formatar_trechos(trechos), pergunta=pergunta.strip())


def remover_think(texto: str) -> tuple[str, bool]:
    """Remove blocos ``<think>...</think>`` (também um bloco aberto e não fechado). Devolve ``(texto, removeu)``."""
    if "<think>" not in texto and "</think>" not in texto:
        return texto, False
    texto = _THINK.sub("", texto)
    if "</think>" in texto:  # fechamento sem abertura: o raciocínio é tudo o que vem antes
        texto = texto.split("</think>", 1)[1]
    return texto, True


def responder(pergunta: str, trechos: list[Trecho], leitor: str, cfg, client: OllamaClient) -> tuple[str, dict]:
    """Gera a resposta do leitor.

    Args:
        pergunta: pergunta do usuário.
        trechos: trechos recuperados (lista vazia em S0).
        leitor: ``base`` ou ``ajustado``.
        cfg: configuração (modelos e decodificação).
        client: cliente Ollama.

    Returns:
        ``(resposta, {'t_leitura_s', 'n_tokens_prompt', 'n_tokens_resposta'})``.
    """
    modelo = cfg.modelo_leitor(leitor)
    prompt = montar_prompt(pergunta, trechos)
    t0 = time.perf_counter()
    r = client.chat(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        options=cfg.decodificacao.opcoes(),
        think=cfg.decodificacao.think,
    )
    t_leitura = time.perf_counter() - t0
    texto, removeu = remover_think(r.get("content", ""))
    if removeu:
        log.warning("bloco <think> removido da resposta de %s apesar de think=%s", modelo, cfg.decodificacao.think)
    return texto.strip(), {
        "t_leitura_s": round(t_leitura, 4),
        "n_tokens_prompt": r.get("prompt_eval_count"),
        "n_tokens_resposta": r.get("eval_count"),
    }
