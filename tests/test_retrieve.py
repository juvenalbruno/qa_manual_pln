from __future__ import annotations

import pymupdf
import pytest

from qa_manual import indexacao
from qa_manual.io_utils import ErroUsuario, ler_json
from qa_manual.retrieve import recuperar, rrf


def test_rrf_ordem_esperada():
    bm25 = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    denso = [("b", 0.9), ("c", 0.8), ("d", 0.7)]
    fundido = rrf([bm25, denso], c=60, k=3)
    assert [i for i, _ in fundido] == ["b", "c", "a"]
    assert fundido[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert len(rrf([bm25, denso], c=60, k=10)) == 4
    # empate desfeito pelo id
    assert [i for i, _ in rrf([[("y", 1)], [("x", 1)]], c=60, k=2)] == ["x", "y"]


def test_modo_nenhum(cfg):
    recuperadas, tempos = recuperar("qualquer pergunta", "nenhum", cfg, None, None)
    assert recuperadas == [] and tempos == {"t_embedding_s": 0.0, "t_busca_s": 0.0}


def test_bm25_traz_o_trecho_com_as_palavras_da_pergunta(cfg, indices):
    pergunta = "Com quantas horas de antecedência o agente marítimo confirma a chegada do navio?"
    recuperadas, _ = recuperar(pergunta, "bm25", cfg, indices.bm25, None)
    assert "72 horas" in indices.por_id[recuperadas[0]["id"]].texto
    assert [r["rank"] for r in recuperadas] == list(range(1, len(recuperadas) + 1))


@pytest.mark.parametrize("modo", ["denso", "hibrido"])
def test_denso_e_hibrido(cfg, indices, modo):
    pergunta = "Em quanto tempo ocorre a liberação da carga após a atracação?"
    recuperadas, tempos = recuperar(pergunta, modo, cfg, indices.bm25, indices.denso)
    assert recuperadas and len(recuperadas) <= cfg.k
    assert "48 horas" in indices.por_id[recuperadas[0]["id"]].texto
    assert tempos["t_embedding_s"] >= 0 and tempos["t_busca_s"] >= 0
    scores = [r["score"] for r in recuperadas]
    assert scores == sorted(scores, reverse=True)


def test_sem_termos_em_comum(cfg, indices):
    assert recuperar("xyzzy plugh", "bm25", cfg, indices.bm25, None)[0] == []


def test_modo_invalido_ou_indice_ausente(cfg, indices):
    with pytest.raises(ErroUsuario):
        recuperar("p", "magico", cfg, indices.bm25, indices.denso)
    with pytest.raises(ErroUsuario):
        recuperar("p", "hibrido", cfg, indices.bm25, None)


def test_manifesto(cfg, indices, pdf_sintetico):
    m = ler_json(indexacao.caminho_manifesto(cfg))
    assert m["n_trechos"] == len(indices.trechos) and m["n_paginas"] == 6
    assert m["modelo_embeddings"] == "bge-m3" and m["dimensao"] == 1024 and len(m["hash_pdf_sha256"]) == 64
    assert m["params_trechos"]["max_tokens"] == 400 and m["bm25"]["k1"] == 1.5


def test_manifesto_bloqueia_pdf_alterado(cfg, indices, pdf_sintetico):
    doc = pymupdf.open(pdf_sintetico)
    doc[0].insert_text((60, 700), "Texto acrescentado depois da indexação.")
    doc.saveIncr()
    doc.close()
    with pytest.raises(ErroUsuario, match="índice desatualizado"):
        indexacao.verificar_manifesto(cfg)


def test_indice_ausente(cfg):
    with pytest.raises(ErroUsuario, match="índice não encontrado"):
        indexacao.verificar_manifesto(cfg)


def test_modelo_de_embeddings_diferente(cfg, indices):
    import dataclasses

    outro = cfg.com(modelos=dataclasses.replace(cfg.modelos, embeddings="outro-modelo"))
    with pytest.raises(ErroUsuario, match="índice desatualizado"):
        indexacao.verificar_manifesto(outro)
