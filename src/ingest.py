"""Etapa 1: extração do PDF, limpeza, detecção de seções e segmentação em passagens."""

from __future__ import annotations

import contextlib
import io
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .utils import ErroUsuario, contar_palavras

MARGEM_CABECALHO = 0.08          # fração da altura da página considerada cabeçalho/rodapé
FRACAO_REPETICAO = 0.3           # linha repetida em >= 30% das páginas é cabeçalho/rodapé
MIN_CARACTERES_POR_PAGINA = 20   # abaixo disso (em média) o PDF é considerado sem camada de texto
MIN_PALAVRAS_SECAO = 25          # seções menores são fundidas à seguinte
SECAO_INICIAL = "Início do documento"
# Avisos comuns em manuais que aparecem em caixa alta/negrito mas não abrem seção.
AVISOS = {"atencao", "cuidado", "aviso", "nota", "notas", "importante", "perigo", "observacao", "obs", "dica", "advertencia"}

_NUMERACAO = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,4})\.?\s+(\S.*)$")
_PAGINA = re.compile(r"^(p[áa]g(ina)?\.?\s*)?\d{1,4}(\s*(de|/|of)\s*\d{1,4})?$", re.IGNORECASE)
_SUMARIO = re.compile(r"(\.\s?){4,}\s*\d+\s*$|…+\s*\d+\s*$")
_MARCADOR_LISTA = re.compile(r"^([•▪●◦\-–*]|\(?[a-z]\)|\d{1,2}[.)])\s+")
_FIM_FRASE = re.compile(r"(?<=[.!?;:])\s+(?=[\"'(«A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9•])")


@dataclass
class Linha:
    texto: str
    pagina: int
    bloco: int
    y0: float
    y1: float
    tamanho: float
    negrito: bool


@dataclass
class Elemento:
    tipo: str          # "titulo" | "paragrafo" | "tabela"
    texto: str
    pagina: int


@dataclass
class Unidade:
    texto: str
    pagina: int
    paragrafo: int
    n: int


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

def _chave_repeticao(texto: str) -> str:
    return re.sub(r"\d+", "#", texto.strip().lower())


def _dentro(bbox: tuple, caixas: list[tuple]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1 for x0, y0, x1, y1 in caixas)


def _extrair_pagina(pagina, num: int, detectar_tabelas: bool) -> tuple[list[Linha], list[tuple[float, str]]]:
    tabelas: list[tuple[float, str]] = []
    caixas: list[tuple] = []
    if detectar_tabelas:
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # silencia dicas impressas pelo PyMuPDF
                encontradas = pagina.find_tables().tables
            for tab in encontradas:
                if tab.row_count < 2 or tab.col_count < 2:
                    continue
                md = tab.to_markdown(clean=True).strip()
                md = unicodedata.normalize("NFKC", md)
                if md:
                    caixas.append(tuple(tab.bbox))
                    tabelas.append((tab.bbox[1], md))
        except Exception:  # detecção de tabelas é best-effort
            tabelas, caixas = [], []

    linhas: list[Linha] = []
    dados = pagina.get_text("dict", sort=True)
    for ib, bloco in enumerate(dados.get("blocks", [])):
        if bloco.get("type") != 0:
            continue
        for ln in bloco.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            if caixas and _dentro(ln["bbox"], caixas):
                continue
            texto = unicodedata.normalize("NFKC", "".join(s["text"] for s in ln["spans"])).strip()
            texto = re.sub(r"\s+", " ", texto)
            negrito = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
            linhas.append(
                Linha(
                    texto=texto,
                    pagina=num,
                    bloco=ib,
                    y0=ln["bbox"][1],
                    y1=ln["bbox"][3],
                    tamanho=round(max(s["size"] for s in spans) * 2) / 2,
                    negrito=bool(negrito),
                )
            )
    return linhas, tabelas


def _remover_cabecalhos(paginas: list[list[Linha]], alturas: list[float]) -> tuple[list[list[Linha]], int]:
    contagem: Counter[str] = Counter()
    for linhas, h in zip(paginas, alturas):
        chaves = {
            _chave_repeticao(l.texto)
            for l in linhas
            if l.y0 < h * MARGEM_CABECALHO or l.y1 > h * (1 - MARGEM_CABECALHO)
        }
        contagem.update(chaves)
    limite = max(2, FRACAO_REPETICAO * len(paginas))
    repetidas = {k for k, c in contagem.items() if c >= limite}

    removidas = 0
    limpas = []
    for linhas, h in zip(paginas, alturas):
        manter = []
        for l in linhas:
            na_margem = l.y0 < h * MARGEM_CABECALHO or l.y1 > h * (1 - MARGEM_CABECALHO)
            if na_margem and (_chave_repeticao(l.texto) in repetidas or _PAGINA.match(l.texto)):
                removidas += 1
                continue
            if _SUMARIO.search(l.texto):  # entradas de sumário ("4.2 Título ....... 17")
                removidas += 1
                continue
            manter.append(l)
        limpas.append(manter)
    return limpas, removidas


def _tamanho_corpo(paginas: list[list[Linha]]) -> float:
    pesos: Counter[float] = Counter()
    for linhas in paginas:
        for l in linhas:
            pesos[l.tamanho] += len(l.texto)
    return pesos.most_common(1)[0][0] if pesos else 10.0


def _caixa_alta(texto: str) -> bool:
    letras = [c for c in texto if c.isalpha()]
    return len(letras) >= 4 and sum(c.isupper() for c in letras) / len(letras) > 0.9


def _sem_acento_min(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto.lower()) if not unicodedata.combining(c))


def eh_titulo(linha: Linha, tamanho_corpo: float) -> bool:
    t = linha.texto.strip()
    palavras = t.split()
    if len(t) < 3 or len(t) > 120 or len(palavras) > 14:
        return False
    if _MARCADOR_LISTA.match(t) and not _NUMERACAO.match(t):
        return False
    if _sem_acento_min(t.rstrip(":!. ")) in AVISOS:
        return False
    maior = linha.tamanho >= tamanho_corpo + 1.0
    termina_frase = t.endswith((".", ",", ";", ":"))
    m = _NUMERACAO.match(t)
    if m:
        resto = m.group(2)
        if not (resto[0].isupper() or resto[0].isdigit()) or len(resto) < 2:
            return False
        niveis = m.group(1).count(".") + 1
        if termina_frase:
            return maior or linha.negrito
        if niveis >= 2:
            return True
        return maior or linha.negrito or _caixa_alta(resto)
    if termina_frase:
        return False
    if maior:
        return True
    if _caixa_alta(t) and len(palavras) <= 10:
        return True
    return linha.negrito and len(palavras) <= 8 and t[0].isupper() and linha.tamanho >= tamanho_corpo


def _juntar_linhas(textos: list[str]) -> str:
    """Junta as linhas de um bloco, desfazendo hifenização e preservando itens de lista."""
    saida = ""
    for t in textos:
        if not saida:
            saida = t
        elif _MARCADOR_LISTA.match(t):
            saida += "\n" + t
        elif re.search(r"[A-Za-zÀ-ÿ]-$", saida) and t[:1].islower():
            saida = saida[:-1] + t
        else:
            saida += " " + t
    return saida


def extrair_elementos(caminho_pdf: Path, detectar_tabelas: bool = True) -> tuple[list[Elemento], dict]:
    try:
        doc = pymupdf.open(caminho_pdf)
    except Exception as e:
        raise ErroUsuario(f"Não foi possível abrir o PDF {caminho_pdf}: {e}") from e
    if doc.needs_pass:
        raise ErroUsuario("O PDF está protegido por senha. Remova a proteção antes de indexar.")

    paginas, alturas, tabelas_pag = [], [], []
    total_chars = 0
    for num, pagina in enumerate(doc, start=1):
        linhas, tabelas = _extrair_pagina(pagina, num, detectar_tabelas)
        paginas.append(linhas)
        alturas.append(pagina.rect.height)
        tabelas_pag.append(tabelas)
        total_chars += sum(len(l.texto) for l in linhas) + sum(len(md) for _, md in tabelas)
    n_paginas = len(paginas)
    doc.close()

    if n_paginas == 0 or total_chars / n_paginas < MIN_CARACTERES_POR_PAGINA:
        raise ErroUsuario(
            "O PDF não possui camada de texto (provavelmente digitalizado). Rode OCR local antes:\n"
            f"  ocrmypdf -l por {caminho_pdf} {caminho_pdf.with_name(caminho_pdf.stem + '_ocr.pdf')}"
        )

    paginas, removidas = _remover_cabecalhos(paginas, alturas)
    corpo = _tamanho_corpo(paginas)

    elementos: list[Elemento] = []
    paginas_sem_texto = 0
    n_tabelas = 0
    for num, (linhas, tabelas) in enumerate(zip(paginas, tabelas_pag), start=1):
        if not linhas and not tabelas:
            paginas_sem_texto += 1
        # Intercala blocos de texto e tabelas pela posição vertical.
        itens: list[tuple[float, str, object]] = [(y, "tabela", md) for y, md in tabelas]
        grupo: list[Linha] = []

        def fechar_grupo():
            if grupo:
                itens.append((grupo[0].y0, "paragrafo", _juntar_linhas([g.texto for g in grupo])))
                grupo.clear()

        for l in linhas:
            if eh_titulo(l, corpo):
                fechar_grupo()
                itens.append((l.y0, "titulo", l.texto))
            else:
                if grupo and grupo[-1].bloco != l.bloco:
                    fechar_grupo()
                grupo.append(l)
        fechar_grupo()
        # sort estável: dentro da mesma página, a ordem de leitura é a de get_text(sort=True);
        # só as tabelas precisam ser reposicionadas.
        textos = [i for i in itens if i[1] != "tabela"]
        tabs = sorted([i for i in itens if i[1] == "tabela"], key=lambda i: i[0])
        n_tabelas += len(tabs)
        mesclados = []
        for item in textos:
            while tabs and tabs[0][0] <= item[0]:
                mesclados.append(tabs.pop(0))
            mesclados.append(item)
        mesclados.extend(tabs)
        for _, tipo, texto in mesclados:
            elementos.append(Elemento(tipo=tipo, texto=str(texto), pagina=num))

    elementos = _juntar_tabelas_continuadas(elementos)
    info = {
        "paginas": n_paginas,
        "paginas_sem_texto": paginas_sem_texto,
        "linhas_cabecalho_rodape_removidas": removidas,
        "tabelas_detectadas": n_tabelas,
        "tamanho_fonte_corpo": corpo,
    }
    return elementos, info


def _colunas(md: str) -> int:
    return md.split("\n", 1)[0].count("|") - 1


def _juntar_tabelas_continuadas(elementos: list[Elemento]) -> list[Elemento]:
    """Une uma tabela à anterior quando ela continua na página seguinte com o mesmo número de colunas.

    A continuação não tem cabeçalho próprio: sua primeira linha (tratada como cabeçalho pelo
    PyMuPDF) volta a ser linha de dados, e a linha separadora "|---|" é descartada.
    """
    saida: list[Elemento] = []
    for el in elementos:
        ant = saida[-1] if saida else None
        if (
            el.tipo == "tabela" and ant is not None and ant.tipo == "tabela"
            and el.pagina == ant.pagina + 1 and _colunas(el.texto) == _colunas(ant.texto)
        ):
            linhas = [l for l in el.texto.split("\n") if not re.fullmatch(r"\|?(\s*:?-+:?\s*\|)+\s*", l)]
            ant.texto = ant.texto + "\n" + "\n".join(linhas)
            continue
        saida.append(el)
    return saida


# ---------------------------------------------------------------------------
# Seções e segmentação
# ---------------------------------------------------------------------------

def agrupar_secoes(elementos: list[Elemento]) -> list[tuple[str, list[Elemento]]]:
    secoes: list[tuple[str, list[Elemento]]] = []
    atual, conteudo = SECAO_INICIAL, []
    for el in elementos:
        if el.tipo == "titulo":
            if conteudo:
                secoes.append((atual, conteudo))
            atual, conteudo = el.texto.strip(), []
        else:
            conteudo.append(el)
    if conteudo:
        secoes.append((atual, conteudo))

    # Funde seções minúsculas na seguinte (ou na anterior, se for a última).
    fundidas: list[tuple[str, list[Elemento]]] = []
    pendente: list[Elemento] = []
    for titulo, conteudo in secoes:
        conteudo = pendente + conteudo
        pendente = []
        if sum(contar_palavras(e.texto) for e in conteudo) < MIN_PALAVRAS_SECAO:
            pendente = conteudo
            continue
        fundidas.append((titulo, conteudo))
    if pendente:
        if fundidas:
            fundidas[-1] = (fundidas[-1][0], fundidas[-1][1] + pendente)
        else:
            fundidas.append((secoes[-1][0] if secoes else SECAO_INICIAL, pendente))
    return fundidas


def _dividir_palavras(texto: str, maximo: int) -> list[str]:
    palavras = texto.split()
    return [" ".join(palavras[i : i + maximo]) for i in range(0, len(palavras), maximo)]


def _unidades(conteudo: list[Elemento], max_palavras: int) -> list[Unidade]:
    unidades: list[Unidade] = []
    for ip, el in enumerate(conteudo):
        if el.tipo == "tabela":
            linhas = el.texto.split("\n")
            cabecalho, corpo = linhas[:2], linhas[2:]
            bloco: list[str] = []
            for linha in corpo or [""]:
                candidato = "\n".join(cabecalho + bloco + [linha])
                if bloco and contar_palavras(candidato) > max_palavras:
                    unidades.append(_unid("\n".join(cabecalho + bloco), el.pagina, ip))
                    bloco = []
                bloco.append(linha)
            texto = "\n".join(cabecalho + bloco).strip()
            if contar_palavras(texto) > max_palavras:
                for parte in _dividir_palavras(texto, max_palavras):
                    unidades.append(_unid(parte, el.pagina, ip))
            elif texto:
                unidades.append(_unid(texto, el.pagina, ip))
            continue
        for item in el.texto.split("\n"):
            for frase in _FIM_FRASE.split(item):
                frase = frase.strip()
                if not frase:
                    continue
                if contar_palavras(frase) > max_palavras:
                    for parte in _dividir_palavras(frase, max_palavras):
                        unidades.append(_unid(parte, el.pagina, ip))
                else:
                    unidades.append(_unid(frase, el.pagina, ip))
    return unidades


def _unid(texto: str, pagina: int, paragrafo: int) -> Unidade:
    return Unidade(texto=texto, pagina=pagina, paragrafo=paragrafo, n=contar_palavras(texto))


def _juntar_unidades(unidades: list[Unidade]) -> str:
    partes = []
    for i, u in enumerate(unidades):
        if i and u.paragrafo == unidades[i - 1].paragrafo and "\n" not in u.texto:
            partes.append(" " + u.texto)
        elif i:
            partes.append("\n" + u.texto)
        else:
            partes.append(u.texto)
    return "".join(partes)


def segmentar_secao(unidades: list[Unidade], max_palavras: int, sobreposicao: int) -> list[list[Unidade]]:
    """Janela gulosa sobre frases: até max_palavras por passagem, com ~sobreposicao palavras repetidas."""
    blocos: list[list[Unidade]] = []
    i = 0
    while i < len(unidades):
        j, n = i, 0
        while j < len(unidades) and (n + unidades[j].n <= max_palavras or j == i):
            n += unidades[j].n
            j += 1
        blocos.append(unidades[i:j])
        if j >= len(unidades):
            break
        volta, acumulado = j, 0
        while (
            volta - 1 > i
            and acumulado < sobreposicao
            and acumulado + unidades[volta - 1].n + unidades[j].n <= max_palavras
        ):
            volta -= 1
            acumulado += unidades[volta].n
        i = volta
    return blocos


def segmentar(
    elementos: list[Elemento], tamanho_tokens: int, sobreposicao_tokens: int, palavras_por_token: float
) -> list[dict]:
    max_palavras = max(20, round(tamanho_tokens * palavras_por_token))
    sobreposicao = max(0, round(sobreposicao_tokens * palavras_por_token))
    passagens: list[dict] = []
    for titulo, conteudo in agrupar_secoes(elementos):
        unidades = _unidades(conteudo, max_palavras)
        for bloco in segmentar_secao(unidades, max_palavras, sobreposicao):
            texto = _juntar_unidades(bloco).strip()
            if not texto:
                continue
            passagens.append(
                {
                    "id": f"p{len(passagens) + 1:04d}",
                    "secao": titulo,
                    "pagina": bloco[0].pagina,
                    "pagina_fim": bloco[-1].pagina,
                    "texto": texto,
                    "n_palavras": contar_palavras(texto),
                }
            )
    return passagens


def estatisticas(passagens: list[dict], info: dict, max_palavras: int) -> dict:
    tamanhos = sorted(p["n_palavras"] for p in passagens) or [0]

    def pct(q: float) -> int:
        return tamanhos[min(len(tamanhos) - 1, int(q * (len(tamanhos) - 1)))]

    faixas = [0, 50, 100, 150, 200, 250, 300, 350, 400, 10**9]
    histograma = {}
    for a, b in zip(faixas, faixas[1:]):
        rotulo = f"{a}-{b - 1}" if b < 10**9 else f">={a}"
        histograma[rotulo] = sum(a <= t < b for t in tamanhos)
    return {
        **info,
        "palavras": sum(tamanhos),
        "passagens": len(passagens),
        "secoes": len({p["secao"] for p in passagens}),
        "limite_palavras_passagem": max_palavras,
        "acima_do_limite": sum(t > max_palavras for t in tamanhos),
        "passagens_vazias": sum(t == 0 for t in tamanhos) if passagens else 0,
        "tamanho_passagem": {
            "min": tamanhos[0],
            "p10": pct(0.1),
            "mediana": statistics.median(tamanhos),
            "media": round(statistics.mean(tamanhos), 1),
            "p90": pct(0.9),
            "max": tamanhos[-1],
        },
        "histograma_palavras": histograma,
    }


def ingerir(caminho_pdf: Path, params: dict) -> tuple[list[dict], dict]:
    elementos, info = extrair_elementos(caminho_pdf, params.get("detectar_tabelas", True))
    passagens = segmentar(
        elementos,
        params["tamanho_tokens"],
        params["sobreposicao_tokens"],
        params["palavras_por_token"],
    )
    if not passagens:
        raise ErroUsuario("Nenhuma passagem foi extraída do PDF.")
    max_palavras = max(20, round(params["tamanho_tokens"] * params["palavras_por_token"]))
    return passagens, estatisticas(passagens, info, max_palavras)
