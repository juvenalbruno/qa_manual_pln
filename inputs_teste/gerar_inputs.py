"""Gera os arquivos de entrada de teste a partir das fontes em inputs_teste/fonte/.

Uso (da raiz do repositório):
    python inputs_teste/gerar_inputs.py

Saídas em inputs_teste/:
    manual_porto_salvador.pdf   manual de teste (conteúdo compilado de fontes públicas)
    gold_dev.jsonl, gold_test.jsonl
    consultas_teste.jsonl       5 consultas para `testar-indice`
    perguntas.txt               perguntas avulsas para `perguntar --arquivo`
    casos_de_erro/              entradas inválidas para testar as mensagens de erro

As evidências do gold (ids pNNNN) correspondem ao índice gerado por `indexar` com os parâmetros
padrão de configs/indexacao.yaml. Cada pergunta de fonte/gold_fonte.yaml tem uma "ancora": um trecho
literal do manual; as passagens que o contêm viram a evidência.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.gold import dividir_por_secao, montar_gold  # noqa: E402
from src.ingest import ingerir  # noqa: E402
from src.utils import FRASE_ABSTENCAO, carregar_config_indexacao, escrever_jsonl  # noqa: E402

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
            if t in (l.strip() for l in doc[n].get_text().splitlines()):
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


def _normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip().lower()


def gerar_gold(passagens: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    fonte = yaml.safe_load((FONTE / "gold_fonte.yaml").read_text(encoding="utf-8"))
    itens, faltando = [], []
    textos = [(p["id"], _normalizar(p["texto"])) for p in passagens]
    for n, g in enumerate(fonte["perguntas"], 1):
        if g["tipo"] == "sem_resposta":
            evid = []
        else:
            ancora = _normalizar(g["ancora"])
            evid = [pid for pid, t in textos if ancora in t]
            if not evid:
                faltando.append(g["ancora"])
        itens.append({
            "origem": f"f{n:03d}",
            "pergunta": g["pergunta"].strip(),
            "resposta_ref": FRASE_ABSTENCAO if g["tipo"] == "sem_resposta" else str(g["resposta"]).strip(),
            "evidencia": evid,
            "tipo": g["tipo"],
            "_consulta_teste": bool(g.get("consulta_teste")),
        })
    if faltando:
        raise SystemExit("Âncoras não encontradas nas passagens:\n  - " + "\n  - ".join(faltando))
    secao_de = {p["id"]: p["secao"] for p in passagens}
    dev, teste = dividir_por_secao(itens, secao_de, frac_dev=0.3, seed=42)
    gold_dev, gold_test = montar_gold(itens, dev, teste)
    return itens, gold_dev, gold_test


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
        "Este arquivo tem extensão .txt; o comando indexar deve recusá-lo.\n", encoding="utf-8")
    escrever_jsonl(destino / "gold_campo_ausente.jsonl", [
        {"id": "q001", "pergunta": "Qual o calado máximo do Porto de Salvador?", "tipo": "factual"},
    ])
    escrever_jsonl(destino / "gold_tipo_invalido.jsonl", [
        {"id": "q001", "pergunta": "Pergunta?", "resposta_ref": "x", "evidencia": [], "tipo": "opinativa"},
    ])
    (destino / "gold_json_quebrado.jsonl").write_text('{"id": "q001", "pergunta": "sem fechar"\n', encoding="utf-8")
    (destino / "gold_vazio.jsonl").write_text("", encoding="utf-8")


def main() -> None:
    gerar_pdf()
    params = carregar_config_indexacao()
    passagens, estat = ingerir(PDF, params)
    itens, gold_dev, gold_test = gerar_gold(passagens)
    escrever_jsonl(DIR / "gold_dev.jsonl", gold_dev)
    escrever_jsonl(DIR / "gold_test.jsonl", gold_test)

    por_origem = {g["origem"]: g for g in gold_dev + gold_test}
    consultas = [{"pergunta": i["pergunta"], "esperada": por_origem[i["origem"]]["evidencia"]}
                 for i in itens if i["_consulta_teste"]]
    escrever_jsonl(DIR / "consultas_teste.jsonl", consultas)

    fonte = yaml.safe_load((FONTE / "gold_fonte.yaml").read_text(encoding="utf-8"))
    (DIR / "perguntas.txt").write_text("\n".join(fonte["perguntas_avulsas"]) + "\n", encoding="utf-8")
    gerar_casos_de_erro(DIR / "casos_de_erro")

    tipos = lambda l: dict(Counter(g["tipo"] for g in l))  # noqa: E731
    secoes = lambda l: {s for g in l for s in (next(p["secao"] for p in passagens if p["id"] == e) for e in g["evidencia"])}  # noqa: E731
    print(f"PDF: {PDF.relative_to(RAIZ)} ({estat['paginas']} páginas, {estat['palavras']} palavras, "
          f"{estat['passagens']} passagens, {estat['secoes']} seções, {estat['tabelas_detectadas']} tabelas)")
    print(f"gold_dev:  {len(gold_dev)} itens {tipos(gold_dev)}")
    print(f"gold_test: {len(gold_test)} itens {tipos(gold_test)}")
    print(f"Seções em comum entre dev e teste: {len(secoes(gold_dev) & secoes(gold_test))}")
    print(f"consultas_teste: {len(consultas)} | perguntas avulsas: {len(fonte['perguntas_avulsas'])}")


if __name__ == "__main__":
    main()
