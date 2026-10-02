"""Consistência dos inputs de teste (``inputs_teste/``) com a ingestão e o agrupamento atuais, sem Ollama.

Se algum teste falhar depois de mudar ``configs/base.yaml``, ``qa_manual.ingest`` ou ``qa_manual.chunking``, rode
``inputs_teste/gerar_inputs.py`` para refazer o gold.
"""

from __future__ import annotations

import pytest

from qa_manual import chunking, cli, gold, indexacao, ingest
from qa_manual.io_utils import RAIZ, ler_jsonl
from qa_manual.ollama_client import OllamaFake
from qa_manual.retrieve import recuperar

DIR = RAIZ / "inputs_teste"
PDF = DIR / "manual_porto_salvador.pdf"
ERROS = DIR / "casos_de_erro"
ARQUIVOS_GOLD = {"dev": DIR / "gold_dev.jsonl", "test": DIR / "gold_test.jsonl"}
CAMPOS = ("id", "pergunta", "resposta_ref", "evidencia", "tipo", "tema_id", "origem", "revisado", "split")
MIN_ITENS = 100
MIN_RECALL_BM25 = 0.8

pytestmark = pytest.mark.skipif(not PDF.is_file(), reason="rode inputs_teste/gerar_inputs.py para gerar o PDF")


@pytest.fixture(scope="module")
def gold_por_split() -> dict[str, list[dict]]:
    return {split: ler_jsonl(caminho) for split, caminho in ARQUIVOS_GOLD.items()}


@pytest.fixture
def trechos(cfg) -> list[chunking.Trecho]:
    return chunking.agrupar_trechos(ingest.construir_paragrafos(PDF, cfg), cfg)


def _todos(gold_por_split: dict[str, list[dict]]) -> list[dict]:
    return [it for itens in gold_por_split.values() for it in itens]


def test_campos_do_contrato(gold_por_split):
    for split, itens in gold_por_split.items():
        for it in itens:
            assert all(c in it for c in CAMPOS), it["id"]
            assert it["split"] == split
            assert it["tipo"] in gold.TIPOS
            assert it["origem"] == "manual" and it["revisado"] is True
            assert isinstance(it["tema_id"], str) and it["tema_id"]
    ids = [it["id"] for it in _todos(gold_por_split)]
    assert len(ids) == len(set(ids))


def test_validar_gold_aceita_os_dois_arquivos(gold_por_split):
    for split, itens in gold_por_split.items():
        gold.validar_gold(itens, ARQUIVOS_GOLD[split])


def test_pelo_menos_100_itens(gold_por_split):
    assert len(_todos(gold_por_split)) >= MIN_ITENS


def test_sem_resposta_se_e_somente_se_sem_evidencia(cfg, gold_por_split):
    for it in _todos(gold_por_split):
        assert (it["tipo"] == "sem_resposta") == (it["evidencia"] == []), it["id"]
        if it["tipo"] == "sem_resposta":
            assert it["resposta_ref"] == cfg.abstencao.frase


def test_nenhum_tema_em_comum_entre_dev_e_test(gold_por_split):
    temas_dev = {it["tema_id"] for it in gold_por_split["dev"]}
    temas_test = {it["tema_id"] for it in gold_por_split["test"]}
    assert temas_dev and temas_test
    assert not temas_dev & temas_test


def test_evidencias_existem_nos_trechos(trechos, gold_por_split):
    por_id = {t.id: t for t in trechos}
    temas = {t.tema_id for t in trechos}
    for it in _todos(gold_por_split):
        assert it["tema_id"] in temas, it["id"]
        for tid in it["evidencia"]:
            assert tid in por_id, f"{it['id']}: {tid} não existe nos trechos"
            assert por_id[tid].tema_id == it["tema_id"], it["id"]


def test_bm25_acha_a_evidencia_no_dev(cfg, gold_por_split):
    client = OllamaFake()
    indexacao.indexar(PDF, cfg, client)
    indices = indexacao.carregar_indices(cfg, client, denso=False)
    cfg_k5 = cfg.com(k=5)
    respondiveis = [it for it in gold_por_split["dev"] if it["evidencia"]]
    acertos = 0
    for it in respondiveis:
        recuperadas, _ = recuperar(it["pergunta"], "bm25", cfg_k5, indices.bm25, None)
        acertos += bool({r["id"] for r in recuperadas} & set(it["evidencia"]))
    assert respondiveis
    assert acertos / len(respondiveis) >= MIN_RECALL_BM25


@pytest.mark.parametrize(
    "args",
    [
        ["indexar", "--manual", str(ERROS / "manual_digitalizado.pdf"), "--permitir-pasta-sincronizada"],
        ["indexar", "--manual", str(ERROS / "manual_extensao_errada.txt")],
        *(
            ["avaliar", "--gold", str(ERROS / nome), "--configs", "S0", "--leitores", "base"]
            for nome in (
                "gold_campo_ausente.jsonl",
                "gold_tipo_invalido.jsonl",
                "gold_json_quebrado.jsonl",
                "gold_vazio.jsonl",
            )
        ),
    ],
    ids=["digitalizado", "extensao", "campo_ausente", "tipo_invalido", "json_quebrado", "vazio"],
)
def test_casos_de_erro_saem_com_codigo_2(cfg, monkeypatch, capsys, args):
    monkeypatch.setattr(cli, "criar_cliente", lambda c: OllamaFake())
    assert cli.main(args) == 2
    assert "erro:" in capsys.readouterr().err
