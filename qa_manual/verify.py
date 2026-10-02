"""Passo 11: detecção de abstenção, extração e validação das citações ``(seção X, p. N)``."""

from __future__ import annotations

import re

from .chunking import Trecho
from .tokenize_pt import remover_acentos

_SECAO = r"se[çc][ãa]o\s*\{?\s*(?P<secao>\d+(?:\.\d+)*)\s*\}?"
_PAGINA = r"(?:p\.|p[áa]g\.?|p[áa]gina)\s*\{?\s*(?P<pagina>\d+)\s*\}?"
_PAR = re.compile(_SECAO + r"\s*[,;–-]?\s*" + _PAGINA, re.IGNORECASE)
_SO_SECAO = re.compile(_SECAO, re.IGNORECASE)
_SO_PAGINA = re.compile(r"(?<![\w.])" + _PAGINA, re.IGNORECASE)
# Parênteses que contêm uma citação: "(seção 4.2, p. 17)", "(fonte: seção 2)", "(p. 3)".
_PARENTESES_CITACAO = re.compile(r"\(\s*[^()]*?(?:se[çc][ãa]o|\bp\.|\bp[áa]g)[^()]*\)", re.IGNORECASE)


def _normalizar_abstencao(t: str) -> str:
    t = remover_acentos(t.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def detectar_abstencao(resposta: str, frase: str) -> bool:
    """True se a resposta contém a frase de abstenção, ignorando maiúsculas, acentos e pontuação."""
    alvo = _normalizar_abstencao(frase)
    return bool(alvo) and alvo in _normalizar_abstencao(resposta)


def extrair_citacoes(resposta: str) -> list[dict]:
    """Extrai citações nos formatos ``(seção 4.2, p. 17)``, ``seção 4.2``, ``p. 17``, ``pág. 17`` e ``página 17``.

    Returns:
        ``[{'tema_id': '4.2', 'pagina': 17}, ...]`` na ordem em que aparecem; o campo ausente fica ``None``.
    """
    achados: list[tuple[int, dict]] = []
    ocupado: list[tuple[int, int]] = []
    for m in _PAR.finditer(resposta):
        achados.append((m.start(), {"tema_id": m.group("secao"), "pagina": int(m.group("pagina"))}))
        ocupado.append(m.span())

    def livre(m: re.Match) -> bool:
        return not any(a <= m.start() < b for a, b in ocupado)

    for m in _SO_SECAO.finditer(resposta):
        if livre(m):
            achados.append((m.start(), {"tema_id": m.group("secao"), "pagina": None}))
    for m in _SO_PAGINA.finditer(resposta):
        if livre(m):
            achados.append((m.start(), {"tema_id": None, "pagina": int(m.group("pagina"))}))
    return [c for _, c in sorted(achados, key=lambda x: x[0])]


def _casa(citacao: dict, t: Trecho) -> bool:
    tid, pag = citacao.get("tema_id"), citacao.get("pagina")
    if tid is None and pag is None:
        return False
    if tid is not None and tid != t.tema_id:
        return False
    return pag is None or t.pagina_inicio <= pag <= t.pagina_fim


def validar_fonte(citacoes: list[dict], recuperadas: list[Trecho]) -> bool:
    """True se alguma citação coincide com um trecho recuperado.

    Coincidir significa: mesma seção (``tema_id``) e página dentro de ``[pagina_inicio, pagina_fim]``. Citações
    parciais (só seção ou só página) são comparadas pelo campo presente. Sem trechos (S0), devolve False.
    """
    return any(_casa(c, t) for c in citacoes for t in recuperadas)


def remover_citacoes(resposta: str) -> str:
    """Resposta sem as citações de fonte, para comparar com a referência (EM, F1, BERTScore)."""
    texto = _PARENTESES_CITACAO.sub(" ", resposta)
    texto = _PAR.sub(" ", texto)
    texto = re.sub(r"\s+([.,;:!?])", r"\1", texto)
    return re.sub(r"\s{2,}", " ", texto).strip(" \n-–")
