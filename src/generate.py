"""Etapa 4: montagem do prompt do leitor e geração da resposta."""

from __future__ import annotations

import re
import time

from .ollama_client import OllamaClient
from .retrieve import Recuperador
from .utils import carregar_prompt, contem_abstencao, renderizar

_CITACAO = re.compile(
    r"[\(\[]?\s*(?:fonte:\s*)?se[çc][ãa]o\s*:?\s*(?P<secao>[^,;()\[\]\n]+?)\s*[,;–-]\s*"
    r"(?:p\.|p[áa]g\.?|p[áa]gina)\s*(?P<pagina>\d+)\s*[\)\]]?\.?",
    re.IGNORECASE,
)


def formatar_passagens(passagens: list[dict]) -> str:
    return "\n".join(
        f"[{i}] (seção {p['secao']}, p. {p['pagina']}) {p['texto']}" for i, p in enumerate(passagens, 1)
    )


def montar_prompt(pergunta: str, passagens: list[dict], cfg: dict) -> str:
    template = carregar_prompt(cfg.get("prompt", "leitor.txt"))
    return renderizar(template, pergunta=pergunta.strip(), passagens=formatar_passagens(passagens))


def extrair_citacao(resposta: str) -> tuple[str, str | None, int | None]:
    """Separa a resposta curta da citação "(seção X, p. N)"; devolve (resposta_curta, secao, pagina)."""
    m = None
    for m in _CITACAO.finditer(resposta):
        pass  # usa a última citação
    curta = _CITACAO.sub("", resposta)
    curta = re.sub(r"^\s*resposta\s*:\s*", "", curta, flags=re.IGNORECASE)
    curta = re.sub(r"\s+([.,;:])", r"\1", curta)
    curta = re.sub(r"\s{2,}", " ", curta).strip(" \n-–")
    if m is None:
        return curta, None, None
    return curta, m.group("secao").strip(), int(m.group("pagina"))


def responder(
    pergunta: str,
    cfg: dict,
    recuperador: Recuperador | None,
    cliente: OllamaClient,
    k: int | None = None,
) -> dict:
    """Recupera as passagens (exceto em S0), gera a resposta e devolve o registro completo."""
    k = int(k or cfg.get("k", 5))
    modo = cfg["modo"]
    t0 = time.perf_counter()
    recuperadas: list[dict] = []
    if modo != "nenhum":
        assert recuperador is not None
        for pid, score in recuperador.recuperar(pergunta, modo, k):
            p = recuperador.indice.por_id[pid]
            recuperadas.append({"id": pid, "score": round(score, 6), "secao": p["secao"], "pagina": p["pagina"]})
    t_recuperacao = time.perf_counter() - t0

    passagens = [recuperador.indice.por_id[r["id"]] for r in recuperadas] if recuperadas else []
    prompt = montar_prompt(pergunta, passagens, cfg)
    saida = cliente.chat(
        cfg["leitor"],
        prompt,
        temperature=cfg.get("temperature", 0),
        seed=cfg.get("seed", 42),
        num_ctx=cfg.get("num_ctx", 8192),
        num_predict=cfg.get("num_predict", 200),
        think=cfg.get("think", False),
    )
    curta, secao_citada, pagina_citada = extrair_citacao(saida.texto)
    return {
        "config": cfg["nome"],
        "modelo": saida.modelo,
        "modo": modo,
        "k": k if modo != "nenhum" else 0,
        "pergunta": pergunta,
        "recuperadas": recuperadas,
        "prompt": prompt,
        "resposta": saida.texto,
        "resposta_curta": curta,
        "secao_citada": secao_citada,
        "pagina_citada": pagina_citada,
        "absteve": contem_abstencao(saida.texto, cfg.get("frases_abstencao", ["Não encontrado no manual"])),
        "latencia_s": round(t_recuperacao + saida.latencia_s, 3),
        "latencia_recuperacao_s": round(t_recuperacao, 3),
        "latencia_geracao_s": round(saida.latencia_s, 3),
        "tokens_prompt": saida.tokens_prompt,
        "tokens_resposta": saida.tokens_resposta,
    }


def aquecer(cliente: OllamaClient, modelo: str, think: bool | None = None) -> None:
    """Carrega o modelo na memória antes de medir latência (a primeira chamada inclui o carregamento)."""
    cliente.chat(modelo, "Olá", num_predict=1, think=think)
