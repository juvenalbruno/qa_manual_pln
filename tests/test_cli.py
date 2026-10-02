"""Fluxo da CLI de ponta a ponta com o cliente Ollama falso."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from qa_manual import cli, metrics
from qa_manual.ollama_client import OllamaFake

from .conftest import leitor_extrativo


def _cliente_para_cli(modelo, prompt):
    if "NÃO aparece" in prompt:
        return json.dumps({"pergunta": "Qual o valor da taxa fictícia de pernoite?"})
    if prompt.startswith("A seguir está um trecho"):
        return json.dumps({"pergunta": "Pergunta gerada?", "resposta": "Resposta gerada."})
    return leitor_extrativo(modelo, prompt)


@pytest.fixture
def fake(monkeypatch):
    client = OllamaFake(resposta=_cliente_para_cli)
    monkeypatch.setattr(cli, "criar_cliente", lambda cfg: client)
    monkeypatch.setattr(metrics, "bertscore_f1", lambda p, r, m, n: [0.7] * len(p))
    return client


def test_ajuda_lista_comandos(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    saida = capsys.readouterr().out
    for comando in (
        "indexar",
        "recuperar",
        "perguntar",
        "gerar-perguntas",
        "exportar-revisao",
        "importar-revisao",
        "perplexidade",
        "avaliar",
        "smoke",
        "stats",
    ):
        assert comando in saida


def test_arquivo_inexistente_sai_com_codigo_2(cfg, fake, capsys):
    assert cli.main(["indexar", "--manual", "nao_existe.pdf"]) == 2
    assert "não encontrado" in capsys.readouterr().err
    assert cli.main(["avaliar", "--gold", "nao_existe.jsonl"]) == 2
    assert cli.main(["perguntar", "--config-exp", "S3", "--pergunta", "x"]) == 2  # sem índice
    assert "rode `qa-manual indexar" in capsys.readouterr().err


def test_pasta_sincronizada_bloqueia_indexacao(cfg, pdf_sintetico, fake, monkeypatch, capsys):
    monkeypatch.setattr(cli, "pasta_sincronizada", lambda p: "iCloud Drive")
    assert cli.main(["indexar", "--manual", str(pdf_sintetico)]) == 2
    assert "sincronizada" in capsys.readouterr().err
    assert cli.main(["indexar", "--manual", str(pdf_sintetico), "--permitir-pasta-sincronizada"]) == 0


def test_fluxo_completo(cfg, pdf_sintetico, fake, capsys):
    assert cli.main(["indexar", "--manual", str(pdf_sintetico), "--reservar-ppl", "2"]) == 0
    assert Path("data/trechos.jsonl").is_file() and Path("index/manifest.json").is_file()
    assert Path("data/ppl_holdout.txt").is_file()
    assert cli.main(["stats"]) == 0
    assert cli.main(["recuperar", "--pergunta", "liberação da carga após a atracação", "--modo", "bm25"]) == 0
    assert "seção 2.2" in capsys.readouterr().out

    assert (
        cli.main(
            [
                "perguntar",
                "--config-exp",
                "S3",
                "--leitor",
                "base",
                "--pergunta",
                "Em quanto tempo ocorre a liberação da carga após a atracação?",
            ]
        )
        == 0
    )
    assert "48 horas" in capsys.readouterr().out
    run = sorted(Path("runs").glob("*_S3_base"))[-1]
    assert (run / "respostas.jsonl").is_file() and (run / "config.json").is_file()
    Path("perguntas.txt").write_text("Qual a velocidade do vento?\n\n# comentário\nQuem fundou o terminal?\n")
    assert cli.main(["perguntar", "--config-exp", "S0", "--leitor", "ajustado", "--arquivo", "perguntas.txt"]) == 0

    assert cli.main(["gerar-perguntas", "--n", "5"]) == 0
    assert cli.main(["gerar-perguntas", "--n", "5"]) == 2  # não sobrescreve sem --forcar
    assert cli.main(["exportar-revisao"]) == 0
    revisao = Path("data/gold_revisao.csv")
    with open(revisao, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    for ln in linhas:
        ln["decisao"] = "aceitar"
    with open(revisao, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
        w.writeheader()
        w.writerows(linhas)
    assert cli.main(["importar-revisao", "--csv", str(revisao), "--dividir", "--reservar-ppl", "2"]) == 0
    assert Path("data/gold_dev.jsonl").is_file() and Path("data/gold_test.jsonl").is_file()

    capsys.readouterr()
    assert (
        cli.main(
            ["avaliar", "--gold", "data/gold_test.jsonl", "--configs", "S0", "S3", "--leitores", "base", "ajustado"]
        )
        == 0
    )
    saida = capsys.readouterr().out
    assert "recall_at_5" in saida and "Ganho pareado" in saida
    assert cli.main(["smoke", "--sem-bertscore"]) == 0
