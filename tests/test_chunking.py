from __future__ import annotations

import dataclasses

from qa_manual.chunking import Trecho, agrupar_trechos, estatisticas, estimar_tokens
from qa_manual.ingest import Paragrafo, construir_paragrafos


def _trechos(cfg, pdf, **mudancas):
    if mudancas:
        cfg = cfg.com(trechos=dataclasses.replace(cfg.trechos, **mudancas))
    ps = construir_paragrafos(pdf, cfg)
    return cfg, ps, agrupar_trechos(ps, cfg)


def test_nenhum_trecho_cruza_tema_e_nenhum_paragrafo_cortado(cfg, pdf_sintetico):
    cfg, ps, ts = _trechos(cfg, pdf_sintetico)
    for p in (p for p in ps if not p.eh_titulo):
        contendo = [t for t in ts if p.texto in t.texto]
        assert len(contendo) == 1, p.idx
        assert contendo[0].tema_id == p.tema_id
    # o texto de cada trecho é exatamente a junção de parágrafos inteiros do mesmo tema
    for t in ts:
        assert all(any(linha == p.texto for p in ps if p.tema_id == t.tema_id) for linha in t.texto.split("\n"))


def test_paragrafo_longo_vira_trecho_sozinho(cfg, pdf_sintetico):
    cfg, _, ts = _trechos(cfg, pdf_sintetico)
    longos = [t for t in ts if t.n_tokens > cfg.trechos.max_tokens]
    assert len(longos) == 1 and longos[0].paragrafos == 1 and longos[0].pagina_inicio == 4


def test_minimo_de_tokens_e_ids_sequenciais(cfg, pdf_sintetico):
    cfg, _, ts = _trechos(cfg, pdf_sintetico, max_tokens=80)
    ultimos = {t.tema_id: t.id for t in ts}  # último trecho de cada tema
    for t in ts:
        assert t.n_tokens >= cfg.trechos.min_tokens or ultimos[t.tema_id] == t.id
    assert [t.id for t in ts] == [f"t{i:04d}" for i in range(1, len(ts) + 1)]
    assert all(t.pagina_inicio <= t.pagina_fim for t in ts)


def _par(idx, tema_id, palavras, pagina=1):
    return Paragrafo(
        idx=idx,
        pagina=pagina,
        tema=f"{tema_id} Tema",
        tema_id=tema_id,
        texto=" ".join(f"p{idx}w{i}" for i in range(palavras)),
        eh_titulo=False,
    )


def test_regras_de_agrupamento(cfg):
    cfg = cfg.com(trechos=dataclasses.replace(cfg.trechos, max_tokens=100, min_tokens=30, tokens_por_palavra=1.0))
    ps = [
        Paragrafo(0, 1, "1 A", "1", "1 A", True),
        _par(1, "1", 10),  # pequeno: fundido com o seguinte
        _par(2, "1", 95),
        _par(3, "1", 50),
        _par(4, "1", 10),  # último do tema: fica pequeno
        Paragrafo(5, 2, "2 B", "2", "2 B", True),
        _par(6, "2", 150, pagina=2),  # maior que max_tokens: sozinho
    ]
    ts = agrupar_trechos(ps, cfg)
    assert [t.paragrafos for t in ts] == [2, 2, 1]
    assert [t.n_tokens for t in ts] == [105, 60, 150]
    assert [t.tema_id for t in ts] == ["1", "1", "2"]


def test_sobreposicao_de_paragrafos(cfg):
    cfg = cfg.com(
        trechos=dataclasses.replace(
            cfg.trechos, max_tokens=60, min_tokens=1, tokens_por_palavra=1.0, sobreposicao_paragrafos=1
        )
    )
    ps = [Paragrafo(0, 1, "1 A", "1", "1 A", True)] + [_par(i, "1", 40) for i in range(1, 4)]
    ts = agrupar_trechos(ps, cfg)
    assert len(ts) == 3
    assert ts[1].texto.split("\n")[0] == ts[0].texto.split("\n")[-1]
    assert ts[0].paragrafos == 1 and ts[1].paragrafos == 2


def test_estatisticas_e_ida_e_volta(cfg, pdf_sintetico):
    cfg, _, ts = _trechos(cfg, pdf_sintetico)
    st = estatisticas(ts, cfg, {"n_paginas": 6})
    assert st["n_trechos"] == len(ts) and st["n_temas"] == 3 and st["n_paginas"] == 6
    assert sum(st["histograma_tokens"].values()) == len(ts) and len(st["histograma_tokens"]) == 10
    assert set(st["trechos_por_tema"]) == {"1", "2.1", "2.2"}
    assert Trecho.de_dict(ts[0].como_dict()) == ts[0]
    assert estimar_tokens("um dois três", 1.3) == 4 and estimar_tokens("", 1.3) == 1
