"""Passo 3: agrupamento de parágrafos em trechos de até ``max_tokens``, sem cruzar temas e sem cortar parágrafos."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from .ingest import Paragrafo

N_FAIXAS_HISTOGRAMA = 10


@dataclass
class Trecho:
    """Unidade de recuperação gravada em ``data/trechos.jsonl`` (contrato 5.1)."""

    id: str
    tema: str
    tema_id: str
    pagina_inicio: int
    pagina_fim: int
    paragrafos: int
    n_tokens: int
    texto: str

    @classmethod
    def de_dict(cls, d: dict) -> Trecho:
        """Constrói a partir de um registro JSONL."""
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def como_dict(self) -> dict:
        """Registro JSONL."""
        return asdict(self)


def estimar_tokens(texto: str, tokens_por_palavra: float) -> int:
    """Estimativa de tokens a partir do número de palavras (1,3 token por palavra em português)."""
    return max(1, round(len(texto.split()) * tokens_por_palavra))


def _tokens(n_palavras: int, tokens_por_palavra: float) -> int:
    return max(1, round(n_palavras * tokens_por_palavra))


def _segmentos_por_tema(paragrafos: list[Paragrafo]) -> list[list[Paragrafo]]:
    """Sequências contíguas de parágrafos (sem títulos) sob o mesmo título."""
    segmentos: list[list[Paragrafo]] = []
    novo = True
    for p in paragrafos:
        if p.eh_titulo:
            novo = True
            continue
        if novo or p.tema_id != segmentos[-1][-1].tema_id or p.tema != segmentos[-1][-1].tema:
            segmentos.append([])
            novo = False
        segmentos[-1].append(p)
    return segmentos


def _agrupar_segmento(seg: list[Paragrafo], max_tokens: int, min_tokens: int, tpp: float) -> list[list[Paragrafo]]:
    # Regras 3 e 4: acumula enquanto couber; um parágrafo maior que max_tokens fica sozinho.
    grupos: list[list[Paragrafo]] = []
    atual: list[Paragrafo] = []
    palavras = 0
    for p in seg:
        n = len(p.texto.split())
        if atual and _tokens(palavras + n, tpp) > max_tokens:
            grupos.append(atual)
            atual, palavras = [], 0
        atual.append(p)
        palavras += n
    if atual:
        grupos.append(atual)

    # Regra 5: grupo abaixo de min_tokens é fundido com o seguinte; o último do tema fica como está.
    fundidos: list[list[Paragrafo]] = []
    pendente: list[Paragrafo] = []
    for i, g in enumerate(grupos):
        g = pendente + g
        pendente = []
        ultimo = i == len(grupos) - 1
        if not ultimo and _tokens(sum(len(p.texto.split()) for p in g), tpp) < min_tokens:
            pendente = g
            continue
        fundidos.append(g)
    return fundidos


def agrupar_trechos(paragrafos: list[Paragrafo], cfg) -> list[Trecho]:
    """Agrupa parágrafos em trechos.

    Regras:
        1. Percorre os parágrafos em ordem; títulos não entram no texto.
        2. Um trecho só contém parágrafos do mesmo tema.
        3. Acumula parágrafos enquanto ``tokens_acumulados + tokens_do_proximo <= max_tokens``.
        4. Um parágrafo maior que ``max_tokens`` vira um trecho sozinho (nunca é cortado).
        5. Um trecho com menos de ``min_tokens`` é fundido com o próximo do mesmo tema; se for o último do
           tema, fica como está.
        6. Se ``sobreposicao_paragrafos`` > 0, cada novo trecho começa repetindo os últimos N parágrafos do
           anterior (mesmo tema).
        7. Ids ``t0001``, ``t0002``, ... na ordem do documento.

    Args:
        paragrafos: saída de :func:`qa_manual.ingest.construir_paragrafos`.
        cfg: :class:`qa_manual.config.Config` (usa ``cfg.trechos``).

    Returns:
        Trechos na ordem do documento; os parágrafos de um trecho são unidos por quebra de linha.
    """
    tc = cfg.trechos
    trechos: list[Trecho] = []
    for seg in _segmentos_por_tema(paragrafos):
        grupos = _agrupar_segmento(seg, tc.max_tokens, tc.min_tokens, tc.tokens_por_palavra)
        n = tc.sobreposicao_paragrafos
        if n > 0:
            grupos = [grupos[0]] + [grupos[i - 1][-n:] + grupos[i] for i in range(1, len(grupos))]
        for g in grupos:
            texto = "\n".join(p.texto for p in g)
            trechos.append(
                Trecho(
                    id=f"t{len(trechos) + 1:04d}",
                    tema=g[0].tema,
                    tema_id=g[0].tema_id,
                    pagina_inicio=min(p.pagina for p in g),
                    pagina_fim=max(p.pagina for p in g),
                    paragrafos=len(g),
                    n_tokens=estimar_tokens(texto, tc.tokens_por_palavra),
                    texto=texto,
                )
            )
    return trechos


def estatisticas(trechos: list[Trecho], cfg, info: dict | None = None) -> dict:
    """Estatísticas de ``data/trechos_stats.json``.

    Contém número de trechos, média/mediana/máximo de tokens, número de temas, trechos por tema (chave =
    numeração do tema, sem o título, para não expor conteúdo) e histograma de tokens em 10 faixas.
    """
    tokens = [t.n_tokens for t in trechos] or [0]
    maximo = max(tokens)
    largura = max(1, -(-maximo // N_FAIXAS_HISTOGRAMA))  # teto
    histograma = Counter(min(t // largura, N_FAIXAS_HISTOGRAMA - 1) for t in tokens)
    faixas = {f"{i * largura}-{(i + 1) * largura - 1}": histograma.get(i, 0) for i in range(N_FAIXAS_HISTOGRAMA)}
    por_tema = Counter(t.tema_id or "sem_numero" for t in trechos)
    return {
        **(info or {}),
        "n_trechos": len(trechos),
        "n_temas": len(por_tema),
        "tokens": {
            "media": round(statistics.mean(tokens), 1),
            "mediana": statistics.median(tokens),
            "minimo": min(tokens),
            "maximo": maximo,
        },
        "acima_de_max_tokens": sum(t > cfg.trechos.max_tokens for t in tokens),
        "abaixo_de_min_tokens": sum(t < cfg.trechos.min_tokens for t in tokens),
        "trechos_por_tema": dict(por_tema),
        "histograma_tokens": faixas,
        "params_trechos": asdict(cfg.trechos),
    }
