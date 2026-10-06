"""Gera os arquivos de entrada de teste a partir das fontes em docs/exemplos/porto_salvador/fonte/.

Uso (da raiz do repositório):
    .venv/bin/python docs/exemplos/porto_salvador/gerar_inputs.py

Saídas em docs/exemplos/porto_salvador/:
    manual_porto_salvador.pdf   manual de teste (conteúdo compilado de fontes públicas)
    gold_dev.jsonl, gold_test.jsonl
    perguntas.txt               perguntas avulsas para `qa-manual perguntar --arquivo`
    casos_de_erro/              entradas inválidas para testar as mensagens de erro (código de saída 2)

As evidências do gold (ids ``tNNNN``) são os trechos que ``qa-manual indexar`` grava com a configuração padrão
(``configs/base.yaml``): o script roda a mesma ingestão (``qa_manual.ingest.construir_paragrafos``) e o mesmo
agrupamento (``qa_manual.chunking.agrupar_trechos``). Cada pergunta respondível de ``fonte/gold_fonte.yaml`` tem
uma ``ancora``, um trecho literal do texto extraído (sem diferenciar maiúsculas e espaços); os trechos que a
contêm viram a evidência, e o ``tema_id`` do item é o desses trechos. Perguntas ``sem_resposta`` não têm
evidência e trazem o ``tema_id`` no próprio YAML. A divisão dev/teste é por tema, com ``qa_manual.gold.dividir``
(``gold.frac_dev`` e ``gold.seed`` da configuração), e nenhum tema aparece nos dois conjuntos.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
import yaml

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))  # permite rodar o script sem `pip install -e .`

from qa_manual import chunking, config, gold, ingest  # noqa: E402
from qa_manual.chunking import Trecho  # noqa: E402
from qa_manual.io_utils import escrever_jsonl  # noqa: E402

DIR = Path(__file__).resolve().parent
FONTE = DIR / "fonte"
PDF = DIR / "manual_porto_salvador.pdf"
CABECALHO = "Manual de Operações Portuárias – Porto de Salvador (documento de teste)"
RODAPE = "Página {n} de {total}"

CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.25; }
h1 { font-size: 16pt; font-weight: bold; margin-top: 16pt; margin-bottom: 6pt; }
h2 { font-size: 12.5pt; font-weight: bold; margin-top: 10pt; margin-bottom: 4pt; }
h3 { font-size: 11pt; font-weight: bold; margin-top: 8pt; margin-bottom: 3pt; }
p { margin-bottom: 5pt; text-align: justify; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; margin-top: 4pt; margin-bottom: 8pt; }
td, th { border: 1px solid black; padding: 2.5pt; font-size: 9pt; }
th { font-weight: bold; }
.capa { font-size: 22pt; font-weight: bold; text-align: center; margin-top: 120pt; }
.subcapa { font-size: 10.5pt; text-align: center; margin-top: 12pt; }
.aviso { font-size: 9pt; text-align: center; margin-top: 60pt; }
.sumario { font-size: 10pt; margin-bottom: 1pt; }
"""


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _renderizar(partes: list[str], saida: Path) -> None:
    """Renderiza cada parte (capa, sumário, corpo) começando em página nova."""
    tmp = saida.with_suffix(".tmp.pdf")
    writer = pymupdf.DocumentWriter(str(tmp))
    pagina = pymupdf.paper_rect("a4")
    area = pagina + (62, 72, -62, -72)
    for html in partes:
        story = pymupdf.Story(html=html, user_css=CSS)
        mais = True
        while mais:
            dispositivo = writer.begin_page(pagina)
            mais, _ = story.place(area)
            story.draw(dispositivo)
            writer.end_page()
    writer.close()

    doc = pymupdf.open(tmp)
    total = doc.page_count
    for n, pg in enumerate(doc, 1):
        if n == 1:
            continue  # capa sem cabeçalho/rodapé
        pg.insert_text((62, 42), CABECALHO, fontsize=7.5)
        pg.insert_text((255, pg.rect.height - 34), RODAPE.format(n=n, total=total), fontsize=7.5)
    doc.save(saida, garbage=3, deflate=True)
    doc.close()
    tmp.unlink()


def _titulos(html: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", t).strip() for t in re.findall(r"<h[12][^>]*>(.*?)</h[12]>", html, re.S)]


def _sumario(titulos: list[str], paginas: dict[str, int]) -> str:
    linhas = []
    for t in titulos:
        recuo = "&#160;" * 4 if re.match(r"^\d+\.\d+", t) else ""
        pontos = "." * max(5, 80 - len(t))  # o pontilhado não pode quebrar a linha
        linhas.append(f'<p class="sumario">{recuo}{t} {pontos} {paginas.get(t, 0)}</p>')
    return "\n".join(linhas)


def _paginas_dos_titulos(pdf: Path, titulos: list[str]) -> dict[str, int]:
    # Linhas do sumário têm pontilhado e número, então só a linha do título no corpo casa exatamente.
    doc = pymupdf.open(pdf)
    paginas = {}
    for t in titulos:
        for n in range(doc.page_count):
            if t in (ln.strip() for ln in doc[n].get_text().splitlines()):
                paginas[t] = n + 1
                break
    doc.close()
    return paginas


def gerar_pdf() -> None:
    capa = (FONTE / "capa.html").read_text(encoding="utf-8")
    corpo = (FONTE / "manual_porto.html").read_text(encoding="utf-8")
    titulos = _titulos(corpo)
    cabecalho_sumario = "<h2>Sumário</h2>\n"
    # Duas passadas: a primeira descobre a página de cada título; a segunda preenche o sumário.
    _renderizar([capa, cabecalho_sumario + _sumario(titulos, {}), corpo], PDF)
    paginas = _paginas_dos_titulos(PDF, titulos)
    _renderizar([capa, cabecalho_sumario + _sumario(titulos, paginas), corpo], PDF)


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------


def _normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip().lower()


def montar_gold(perguntas: list[dict], trechos: list[Trecho], frase_abstencao: str) -> list[dict]:
    """Converte as perguntas de ``gold_fonte.yaml`` em itens do gold (sem o campo ``split``).

    Raises:
        SystemExit: com a lista de âncoras não encontradas ou ambíguas e de ``tema_id`` inválidos.
    """
    textos = [(t, _normalizar(t.texto)) for t in trechos]
    temas = {t.tema_id for t in trechos if t.tema_id}
    itens, problemas = [], []
    for n, g in enumerate(perguntas, 1):
        tipo = g.get("tipo")
        rotulo = f"q{n:03d} ({g.get('pergunta', '?')})"
        if tipo not in gold.TIPOS:
            problemas.append(f"{rotulo}: tipo '{tipo}' inválido; use {', '.join(gold.TIPOS)}")
            continue
        if tipo == "sem_resposta":
            evidencia, tema_id = [], g.get("tema_id")
            if not isinstance(tema_id, str) or tema_id not in temas:
                problemas.append(f"{rotulo}: sem_resposta exige tema_id existente nos trechos, entre aspas")
            resposta = frase_abstencao
        else:
            ancora = _normalizar(g.get("ancora", ""))
            achados = [t for t, texto in textos if ancora and ancora in texto]
            temas_achados = sorted({t.tema_id for t in achados})
            if not achados:
                problemas.append(f"{rotulo}: âncora não encontrada nos trechos: {g.get('ancora')!r}")
            elif len(temas_achados) > 1:
                problemas.append(f"{rotulo}: âncora em temas diferentes ({', '.join(temas_achados)}): {g['ancora']!r}")
            evidencia = [t.id for t in achados]
            tema_id = achados[0].tema_id if achados else ""
            resposta = str(g["resposta"]).strip()
        itens.append(
            {
                "id": f"q{n:03d}",
                "pergunta": g["pergunta"].strip(),
                "resposta_ref": resposta,
                "evidencia": evidencia,
                "tipo": tipo,
                "tema_id": tema_id,
                "origem": "manual",
                "revisado": True,
            }
        )
    if problemas:
        raise SystemExit("Problemas em fonte/gold_fonte.yaml:\n  - " + "\n  - ".join(problemas))
    return itens


# ---------------------------------------------------------------------------
# Casos de erro
# ---------------------------------------------------------------------------


def gerar_casos_de_erro(destino: Path) -> None:
    destino.mkdir(exist_ok=True)
    # PDF "digitalizado": páginas rasterizadas, sem camada de texto.
    origem = pymupdf.open(PDF)
    scan = pymupdf.open()
    for n in range(min(3, origem.page_count)):
        pix = origem[n].get_pixmap(dpi=60)
        pg = scan.new_page(width=origem[n].rect.width, height=origem[n].rect.height)
        pg.insert_image(pg.rect, pixmap=pix)
    scan.save(destino / "manual_digitalizado.pdf", deflate=True)
    scan.close()
    origem.close()
    (destino / "manual_extensao_errada.txt").write_text(
        "Este arquivo tem extensão .txt; o comando indexar deve recusá-lo.\n", encoding="utf-8"
    )
    escrever_jsonl(
        destino / "gold_campo_ausente.jsonl",
        [{"id": "q001", "pergunta": "Qual o calado máximo do Porto de Salvador?", "tipo": "factual"}],
    )
    escrever_jsonl(
        destino / "gold_tipo_invalido.jsonl",
        [{"id": "q001", "pergunta": "Pergunta?", "resposta_ref": "x", "evidencia": [], "tipo": "opinativa"}],
    )
    (destino / "gold_json_quebrado.jsonl").write_text('{"id": "q001", "pergunta": "sem fechar"\n', encoding="utf-8")
    (destino / "gold_vazio.jsonl").write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = config.carregar()
    gerar_pdf()
    info: dict = {}
    paragrafos = ingest.construir_paragrafos(PDF, cfg, info)
    trechos = chunking.agrupar_trechos(paragrafos, cfg)

    fonte = yaml.safe_load((FONTE / "gold_fonte.yaml").read_text(encoding="utf-8"))
    itens = montar_gold(fonte["perguntas"], trechos, cfg.abstencao.frase)
    gold_dev, gold_test = gold.dividir(itens, cfg.gold.frac_dev, cfg.gold.seed)
    gold.validar_gold(gold_dev, "gold_dev")
    gold.validar_gold(gold_test, "gold_test")
    comuns = {g["tema_id"] for g in gold_dev} & {g["tema_id"] for g in gold_test}
    if comuns:
        raise SystemExit(f"Temas em comum entre dev e teste: {', '.join(sorted(comuns))}")
    escrever_jsonl(DIR / "gold_dev.jsonl", gold_dev)
    escrever_jsonl(DIR / "gold_test.jsonl", gold_test)

    (DIR / "perguntas.txt").write_text("\n".join(fonte["perguntas_avulsas"]) + "\n", encoding="utf-8")
    gerar_casos_de_erro(DIR / "casos_de_erro")

    def resumo(lista: list[dict]) -> str:
        tipos = Counter(g["tipo"] for g in lista)
        return f"{len(lista)} itens, {len({g['tema_id'] for g in lista})} temas {dict(sorted(tipos.items()))}"

    print(
        f"PDF: {PDF.relative_to(RAIZ)} ({info['n_paginas']} páginas, {len(trechos)} trechos, "
        f"{len({t.tema_id for t in trechos})} temas, {info['paragrafos_curtos_descartados']} blocos curtos descartados)"
    )
    print(f"gold_dev:  {resumo(gold_dev)}")
    print(f"gold_test: {resumo(gold_test)}")
    print(f"Temas em comum entre dev e teste: {len(comuns)} | perguntas avulsas: {len(fonte['perguntas_avulsas'])}")


if __name__ == "__main__":
    main()
