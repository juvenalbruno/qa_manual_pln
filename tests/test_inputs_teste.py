"""Os arquivos de inputs_teste/ devem ser consistentes com o índice gerado pelo comando indexar."""

import shutil
from pathlib import Path

import pytest

from src.cli import main
from src.utils import ler_json, ler_jsonl

INPUTS = Path(__file__).resolve().parent.parent / "inputs_teste"
pytestmark = pytest.mark.skipif(not (INPUTS / "manual_porto_salvador.pdf").exists(),
                                reason="rode inputs_teste/gerar_inputs.py antes")


@pytest.fixture
def inputs(tmp_path, monkeypatch, fake_ollama):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_HOST", fake_ollama.url)
    destino = tmp_path / "inputs_teste"
    shutil.copytree(INPUTS, destino, ignore=shutil.ignore_patterns("fonte", "*.py"))
    return destino


def test_evidencias_correspondem_ao_indice(inputs, capsys):
    assert main(["indexar", "--manual", "inputs_teste/manual_porto_salvador.pdf"]) == 0
    passagens = {p["id"]: p for p in ler_jsonl("index/passages.jsonl")}
    gold = ler_jsonl(inputs / "gold_dev.jsonl") + ler_jsonl(inputs / "gold_test.jsonl")
    assert len(gold) >= 100
    for g in gold:
        assert all(e in passagens for e in g["evidencia"]), g["id"]
        assert (g["tipo"] == "sem_resposta") == (not g["evidencia"])
    secoes = lambda split: {passagens[e]["secao"] for g in gold if g["split"] == split for e in g["evidencia"]}  # noqa: E731
    assert not secoes("dev") & secoes("test")
    assert ler_json("index/stats.json")["acima_do_limite"] == 0

    capsys.readouterr()
    assert main(["testar-indice", "--consultas", "inputs_teste/consultas_teste.jsonl"]) == 0
    assert "bm25: 5/5" in capsys.readouterr().out

    assert main(["avaliar", "--gold", "inputs_teste/gold_dev.jsonl", "--configs", "S1", "--sem-juiz"]) == 0
    run = sorted(Path("runs").glob("*_avaliar"))[-1]
    assert ler_json(run / "metrics.json")["S1"]["recall@5"] >= 0.8
    assert main(["perguntar", "--config", "S3", "--arquivo", "inputs_teste/perguntas.txt"]) == 0


def test_executar_pipeline_completo(inputs, capsys):
    args = ["executar",
            "--manual", "inputs_teste/manual_porto_salvador.pdf",
            "--gold", "inputs_teste/gold_test.jsonl",
            "--gold-dev", "inputs_teste/gold_dev.jsonl",
            "--consultas", "inputs_teste/consultas_teste.jsonl",
            "--perguntas", "inputs_teste/perguntas.txt",
            "--repeticoes", "2", "--varrer", "--tamanhos", "300", "400", "--ks", "3", "5"]
    assert main(args) == 0
    run = sorted(Path("runs").glob("*_avaliar"))[-1]
    for arq in ("metrics.json", "analise.md", "recall_k.png", "comparacoes.json", "erros_amostra.jsonl"):
        assert (run / arq).exists(), arq
    assert set(ler_json(run / "metrics.json")) >= {"S0", "S1", "S2", "S3"}
    assert sorted(Path("runs").glob("*_varredura"))
    assert sorted(Path("runs").glob("*_perguntar_S3"))
    assert "Resumo" in capsys.readouterr().out
    # segunda execução reaproveita o índice
    assert main(args[:5] + ["--configs", "S1", "--sem-juiz"]) == 0
    assert "já está atualizado" in capsys.readouterr().out


def test_executar_valida_arquivos_antes_de_comecar(inputs):
    assert main(["executar", "--manual", "inputs_teste/manual_porto_salvador.pdf",
                 "--gold", "inputs_teste/nao_existe.jsonl"]) == 2
    assert main(["executar", "--manual", "inputs_teste/manual_porto_salvador.pdf",
                 "--gold", "inputs_teste/gold_test.jsonl", "--varrer"]) == 2  # varrer sem --gold-dev
    assert not Path("index").exists()


@pytest.mark.parametrize("args", [
    ["indexar", "--manual", "inputs_teste/casos_de_erro/manual_digitalizado.pdf"],
    ["indexar", "--manual", "inputs_teste/casos_de_erro/manual_extensao_errada.txt"],
    ["avaliar", "--gold", "inputs_teste/casos_de_erro/gold_campo_ausente.jsonl", "--configs", "S0"],
    ["avaliar", "--gold", "inputs_teste/casos_de_erro/gold_tipo_invalido.jsonl", "--configs", "S0"],
    ["avaliar", "--gold", "inputs_teste/casos_de_erro/gold_json_quebrado.jsonl", "--configs", "S0"],
    ["avaliar", "--gold", "inputs_teste/casos_de_erro/gold_vazio.jsonl", "--configs", "S0"],
])
def test_casos_de_erro(inputs, args):
    assert main(args) == 2
