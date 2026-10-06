from __future__ import annotations

import pymupdf
import pytest

from qa_manual.ingest import (
    TEMA_INICIAL,
    construir_paragrafos,
    detectar_cabecalhos_rodapes,
    eh_titulo,
    extrair_blocos,
    limpar_texto,
    titulo_plausivel,
)
from qa_manual.io_utils import ErroUsuario

from .conftest import CABECALHO


def test_cabecalho_e_rodape_removidos(cfg, pdf_sintetico):
    ps = construir_paragrafos(pdf_sintetico, cfg)
    assert any(CABECALHO in t for _, t in extrair_blocos(pdf_sintetico))
    assert not any(CABECALHO in p.texto or "Página" in p.texto for p in ps)


def test_hifenizacao_corrigida(cfg, pdf_sintetico):
    ps = construir_paragrafos(pdf_sintetico, cfg)
    alvo = [p for p in ps if "liberação de novas versões" in p.texto]
    assert len(alvo) == 1 and alvo[0].pagina == 2
    assert not any("libera-" in p.texto for p in ps)


def test_titulos_com_tema_id_e_pagina(cfg, pdf_sintetico):
    info: dict = {}
    ps = construir_paragrafos(pdf_sintetico, cfg, info)
    titulos = [(p.tema_id, p.tema, p.pagina) for p in ps if p.eh_titulo]
    assert titulos == [
        ("1", "1 Introdução", 1),
        ("2.1", "2.1 Atracação", 3),
        ("2.2", "2.2 Liberação de carga", 5),
    ]
    assert info["n_paginas"] == 6 and info["n_titulos"] == 3
    # tema propagado e páginas corretas
    atracacao = [p for p in ps if "72 horas" in p.texto]
    assert atracacao[0].tema_id == "2.1" and atracacao[0].pagina == 3
    longo = [p for p in ps if len(p.texto.split()) >= 700]
    assert len(longo) == 1 and longo[0].pagina == 4 and longo[0].tema_id == "2.1"
    assert [p.idx for p in ps] == list(range(len(ps)))


def test_limpar_texto():
    assert limpar_texto("A libera-\nção da carga") == "A liberação da carga"
    assert limpar_texto("linha um\nlinha dois\n\nnovo  parágrafo\x07") == "linha um linha dois\n\nnovo parágrafo"
    assert limpar_texto("ﬁscalização\t aduaneira") == "fiscalização aduaneira"
    assert limpar_texto("até 15 km/\nh no pátio") == "até 15 km/h no pátio"
    assert limpar_texto("Ex-\nPresidente") == "Ex- Presidente"  # maiúscula: não é hifenização


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("4.2 Liberação de carga", (True, "4.2", "4.2 Liberação de carga")),
        ("1 Introdução", (True, "1", "1 Introdução")),
        ("2.5 kg de carga máxima", (False, "", "")),
        ("1. Abra a tampa", (False, "", "")),
        ("Texto comum sem número", (False, "", "")),
    ],
)
def test_eh_titulo(cfg, texto, esperado):
    assert eh_titulo(texto, cfg.ingestao.regex_titulo) == esperado


def test_titulo_plausivel():
    assert titulo_plausivel("1", "")
    assert titulo_plausivel("2.4", "2.3") and titulo_plausivel("3", "2.3") and titulo_plausivel("2.3.1", "2.3")
    assert not titulo_plausivel("1", "13.1")  # passo numerado de procedimento
    assert not titulo_plausivel("201", "2.3")  # linha de tabela
    assert titulo_plausivel("2.1", "1") and titulo_plausivel("3.1", "2.3")  # capítulo sem título próprio
    assert not titulo_plausivel("2.3", "2.3") and not titulo_plausivel("2", "2.3")
    assert not titulo_plausivel("2.9", "2.3") and not titulo_plausivel("2.4.7", "2.3")


def test_passos_numerados_nao_viram_secao(cfg, tmp_path):
    doc = pymupdf.open()
    pg = doc.new_page()
    html = (
        "<h1>3 Emergências</h1><p>Em caso de emergência, o colaborador segue os passos abaixo, nesta ordem.</p>"
        "<p>1 Acionar o alarme mais próximo do local.</p><p>2 Comunicar a central de segurança.</p>"
        "<h1>4 Manutenção</h1><p>Os equipamentos passam por manutenção preventiva periódica.</p>"
    )
    pg.insert_htmlbox(pymupdf.Rect(50, 50, 550, 800), html)
    destino = tmp_path / "passos.pdf"
    doc.save(destino)
    ps = construir_paragrafos(destino, cfg)
    assert [p.tema_id for p in ps if p.eh_titulo] == ["3", "4"]
    assert any(p.texto.startswith("1 Acionar") and p.tema_id == "3" for p in ps)


def test_detectar_cabecalhos_rodapes():
    blocos = [(1, "Cabeçalho 1"), (2, "Cabeçalho 2"), (3, "Cabeçalho 3"), (1, "texto único"), (2, "17")]
    repetidos = detectar_cabecalhos_rodapes(blocos, 3, 0.5)
    assert "Cabeçalho" in repetidos and "texto único" not in repetidos


def test_texto_antes_do_primeiro_titulo(cfg, tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_htmlbox(
        pymupdf.Rect(50, 50, 550, 800),
        "<p>Apresentação geral do documento fictício antes de qualquer seção numerada.</p><h1>1 Escopo</h1>"
        "<p>O escopo cobre as operações do terminal fictício usado nos testes automatizados.</p>",
    )
    destino = tmp_path / "pre.pdf"
    doc.save(destino)
    ps = construir_paragrafos(destino, cfg)
    assert ps[0].tema == TEMA_INICIAL and ps[0].tema_id == ""


def test_pdf_sem_texto(cfg, tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        pg = doc.new_page()
        pg.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0.5, 0.5, 0.5))
    destino = tmp_path / "scan.pdf"
    doc.save(destino)
    with pytest.raises(ErroUsuario, match="PDF sem texto; rode OCR local"):
        construir_paragrafos(destino, cfg)


def test_arquivo_que_nao_e_pdf(cfg, tmp_path):
    falso = tmp_path / "falso.pdf"
    falso.write_text("não sou um PDF")
    with pytest.raises(ErroUsuario):
        construir_paragrafos(falso, cfg)
