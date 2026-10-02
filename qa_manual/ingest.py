"""Passos 1 e 2: texto do PDF por página (PyMuPDF) e parágrafos com tema e página.

Fluxo: blocos de texto (``page.get_text("blocks")``) -> remoção de cabeçalhos/rodapés repetidos -> limpeza
(hifenização, quebras de linha) -> detecção de títulos pela regex configurada -> propagação do tema corrente ->
descarte de blocos curtos. Títulos viram parágrafos com ``eh_titulo=True``: só definem o tema e não entram no
texto dos trechos.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .io_utils import ErroUsuario

log = logging.getLogger(__name__)

TEMA_INICIAL = "Sem título"
MIN_CHARS_PAGINA = 50  # página com menos caracteres que isso conta como "sem texto"
FRAC_PAGINAS_SEM_TEXTO = 0.8  # acima dessa fração de páginas sem texto, o PDF precisa de OCR
MAX_SALTO_NUMERACAO = 5  # maior salto aceito entre seções irmãs (2.3 -> 2.8)
# Entrada de sumário: "4.2 Liberação de carga ........ 17". Linhas assim não são títulos nem conteúdo.
_SUMARIO = re.compile(r"(\.\s?){4,}\s*\d+\s*$|…+\s*\d+\s*$")
_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Só ligaduras tipográficas: NFKC também trocaria "nº" por "no" e "m³" por "m3".
_LIGADURAS = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})


@dataclass
class Paragrafo:
    """Parágrafo extraído do PDF, já associado ao tema (seção) corrente."""

    idx: int
    pagina: int
    tema: str
    tema_id: str
    texto: str
    eh_titulo: bool


def _abrir(pdf_path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise ErroUsuario(f"Não foi possível abrir o PDF {pdf_path}: {type(e).__name__}.") from None
    if not doc.is_pdf:
        doc.close()
        raise ErroUsuario(f"{pdf_path} não é um PDF.")
    if doc.needs_pass:
        doc.close()
        raise ErroUsuario(f"{pdf_path} está protegido por senha; remova a proteção antes de indexar.")
    return doc


def _extrair(pdf_path: Path) -> tuple[list[tuple[int, str]], int]:
    blocos: list[tuple[int, str]] = []
    with _abrir(pdf_path) as doc:
        n_paginas = doc.page_count
        for num, pagina in enumerate(doc, start=1):
            # (x0, y0, x1, y1, texto, numero_bloco, tipo); tipo 1 = imagem
            texto = [b for b in pagina.get_text("blocks") if b[6] == 0 and b[4].strip()]
            texto.sort(key=lambda b: (round(b[1], 1), b[0]))
            blocos.extend((num, b[4]) for b in texto)
    return blocos, n_paginas


def extrair_blocos(pdf_path: Path) -> list[tuple[int, str]]:
    """Blocos de texto de todas as páginas, na ordem de leitura (y, depois x).

    Args:
        pdf_path: caminho do PDF.

    Returns:
        Lista de ``(pagina, texto_do_bloco)`` com página 1-based. Blocos de imagem são ignorados.
    """
    return _extrair(Path(pdf_path))[0]


def _chave_repeticao(texto: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d", "", texto.strip())).strip()


def detectar_cabecalhos_rodapes(blocos: list[tuple[int, str]], n_paginas: int, limiar: float) -> set[str]:
    """Identifica cabeçalhos e rodapés pela repetição entre páginas.

    Cada bloco é normalizado (strip, remoção de dígitos, espaços colapsados) e contado uma vez por página. Os
    que aparecem em mais de ``limiar * n_paginas`` páginas (e em pelo menos duas) são cabeçalho ou rodapé.

    Returns:
        Conjunto de chaves normalizadas a remover (use a mesma normalização para filtrar).
    """
    paginas_por_chave: dict[str, set[int]] = defaultdict(set)
    for pagina, texto in blocos:
        paginas_por_chave[_chave_repeticao(texto)].add(pagina)
    return {
        chave for chave, paginas in paginas_por_chave.items() if len(paginas) > limiar * n_paginas and len(paginas) >= 2
    }


def limpar_texto(t: str) -> str:
    """Limpa o texto de um bloco.

    Junta hifenização de fim de linha (``libera-\\nção`` -> ``liberação``), troca quebras simples por espaço,
    mantém quebras duplas, colapsa espaços, remove caracteres de controle e desfaz ligaduras tipográficas.
    """
    t = unicodedata.normalize("NFC", t).translate(_LIGADURAS)
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").replace("\t", " ")
    t = t.replace("\xad\n", "").replace("\xad", "")
    t = _CONTROLE.sub("", t)
    t = re.sub(r"(\w)-[ ]*\n[ ]*(?=[a-zà-öø-ÿ])", r"\1", t)
    partes = re.split(r"\n[ ]*\n", t)
    partes = [re.sub(r" {2,}", " ", re.sub(r"[ ]*\n[ ]*", " ", p)).strip() for p in partes]
    return "\n\n".join(p for p in partes if p)


def eh_titulo(t: str, regex: str) -> tuple[bool, str, str]:
    """Aplica a regex de título.

    Args:
        t: texto (uma linha) candidato a título.
        regex: expressão com a numeração no grupo 1, por exemplo ``^(\\d+(\\.\\d+){0,3})\\s+[A-Z]...``.

    Returns:
        ``(True, tema_id, tema)`` se casar, com ``tema_id`` = numeração (``"4.2"``) e ``tema`` = título
        completo; senão ``(False, "", "")``.
    """
    t = t.strip()
    m = re.match(regex, t)
    if not m:
        return False, "", ""
    return True, (m.group(1) if m.groups() else "") or "", t


def _partes(bruto: str, regex: str) -> list[tuple[tuple[str, str] | None, list[str], list[str]]]:
    """Divide um bloco nas quebras duplas e separa um eventual título no início de cada parte.

    Entradas de sumário são descartadas. O título é a primeira linha que casa com a regex; se o bloco tiver só
    duas linhas e as duas juntas também casarem, é um título quebrado em duas linhas.

    Returns:
        ``[((tema_id, tema) ou None, linhas_do_resto, todas_as_linhas)]``.
    """
    saida = []
    for parte in re.split(r"\n\s*\n", bruto):
        linhas = [ln.strip() for ln in parte.split("\n") if ln.strip()]
        linhas = [ln for ln in linhas if not _SUMARIO.search(ln)]
        if not linhas:
            continue
        ok, tema_id, tema = eh_titulo(limpar_texto(linhas[0]), regex)
        if not ok:
            saida.append((None, linhas, linhas))
            continue
        if len(linhas) == 2:
            ok2, tid2, tema2 = eh_titulo(limpar_texto("\n".join(linhas)), regex)
            if ok2:
                saida.append(((tid2, tema2), [], linhas))
                continue
        saida.append(((tema_id, tema), linhas[1:], linhas))
    return saida


def _numeracao(tema_id: str) -> tuple[int, ...]:
    return tuple(int(x) for x in tema_id.split(".") if x.isdigit())


def titulo_plausivel(novo: str, atual: str | None, max_salto: int = MAX_SALTO_NUMERACAO) -> bool:
    """Confere se a numeração ``novo`` pode suceder a seção ``atual``.

    Aceita avançar em algum nível com salto de até ``max_salto`` (``2.3`` -> ``2.4``, ``3``, ``3.1``) ou descer
    para subseções (``2.3`` -> ``2.3.1``); níveis abertos depois do ponto de avanço devem começar baixos
    (até ``max_salto``), o que admite capítulos sem título próprio (``1`` -> ``2.1``). Rejeita, por exemplo,
    passos numerados (``13.1`` -> ``1``) e linhas de tabela (``2.3`` -> ``201``). O primeiro título do
    documento é sempre aceito.
    """
    if not atual:
        return True
    n, a = _numeracao(novo), _numeracao(atual)
    if not n or not a:
        return False
    comum = 0
    while comum < min(len(n), len(a)) and n[comum] == a[comum]:
        comum += 1
    if comum == len(n):  # repete a seção atual ou um ancestral dela
        return False
    if comum < len(a) and not 0 < n[comum] - a[comum] <= max_salto:
        return False
    inicio = comum + 1 if comum < len(a) else comum
    return all(1 <= x <= max_salto for x in n[inicio:])


def _verificar_camada_texto(blocos: list[tuple[int, str]], n_paginas: int) -> None:
    chars = defaultdict(int)
    for pagina, texto in blocos:
        chars[pagina] += len(texto.strip())
    sem_texto = sum(1 for p in range(1, n_paginas + 1) if chars[p] < MIN_CHARS_PAGINA)
    if n_paginas == 0 or sem_texto > FRAC_PAGINAS_SEM_TEXTO * n_paginas:
        raise ErroUsuario("PDF sem texto; rode OCR local (ocrmypdf) antes")


def construir_paragrafos(pdf_path: Path, cfg, info: dict | None = None) -> list[Paragrafo]:
    """Passos 1 e 2 completos: PDF -> parágrafos com tema e página.

    Args:
        pdf_path: caminho do PDF.
        cfg: :class:`qa_manual.config.Config` (usa ``cfg.ingestao``).
        info: se informado, recebe contagens da extração (páginas, blocos removidos, títulos etc.).

    Returns:
        Parágrafos na ordem do documento, incluindo os títulos (``eh_titulo=True``).

    Raises:
        ErroUsuario: se o PDF não abrir ou não tiver camada de texto.
    """
    ing = cfg.ingestao
    blocos, n_paginas = _extrair(Path(pdf_path))
    _verificar_camada_texto(blocos, n_paginas)

    n_blocos = len(blocos)
    if ing.remover_cabecalhos_rodapes:
        repetidos = detectar_cabecalhos_rodapes(blocos, n_paginas, ing.limiar_repeticao_cabecalho)
        blocos = [(p, t) for p, t in blocos if _chave_repeticao(t) not in repetidos]
    removidos = n_blocos - len(blocos)

    paragrafos: list[Paragrafo] = []
    tema, tema_id = TEMA_INICIAL, ""
    curtos = rejeitados = 0

    def adicionar(pagina: int, texto: str, titulo: bool) -> None:
        paragrafos.append(
            Paragrafo(idx=len(paragrafos), pagina=pagina, tema=tema, tema_id=tema_id, texto=texto, eh_titulo=titulo)
        )

    for pagina, bruto in blocos:
        for titulo, resto, linhas in _partes(bruto, ing.regex_titulo):
            if titulo and ing.validar_sequencia_titulos and not titulo_plausivel(titulo[0], tema_id):
                rejeitados += 1
                titulo, resto = None, linhas
            if titulo:
                tema_id, tema = titulo
                adicionar(pagina, tema, True)
            for texto in limpar_texto("\n".join(resto)).split("\n\n"):
                if len(texto) < ing.min_chars_paragrafo:
                    curtos += bool(texto)
                    continue
                adicionar(pagina, texto, False)

    n_titulos = sum(p.eh_titulo for p in paragrafos)
    log.info(
        "ingestão: %d páginas, %d blocos, %d cabeçalhos/rodapés removidos, %d títulos (%d candidatos fora de "
        "sequência tratados como texto), %d parágrafos, %d curtos descartados",
        n_paginas,
        n_blocos,
        removidos,
        n_titulos,
        rejeitados,
        len(paragrafos) - n_titulos,
        curtos,
    )
    if info is not None:
        info.update(
            n_paginas=n_paginas,
            n_blocos=n_blocos,
            blocos_cabecalho_rodape_removidos=removidos,
            n_titulos=n_titulos,
            titulos_fora_de_sequencia=rejeitados,
            n_paragrafos=len(paragrafos) - n_titulos,
            paragrafos_curtos_descartados=curtos,
        )
    if not any(not p.eh_titulo for p in paragrafos):
        raise ErroUsuario(f"Nenhum parágrafo extraído de {pdf_path}; confira o PDF e ingestao.min_chars_paragrafo.")
    return paragrafos
