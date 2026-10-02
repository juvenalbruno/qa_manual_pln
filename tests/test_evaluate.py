from __future__ import annotations

import math

import pandas as pd
import pytest

from qa_manual import config
from qa_manual.evaluate import avaliar, bertscore_com_cache, localizar_perplexidade
from qa_manual.io_utils import ErroUsuario, escrever_json, escrever_jsonl, ler_jsonl
from qa_manual.ollama_client import ModeloAusente, OllamaFake

from .conftest import leitor_extrativo

COLUNAS_5_7 = [
    "config",
    "leitor",
    "n",
    "em",
    "f1_tokens",
    "bertscore_f1",
    "recall_at_5",
    "mrr_at_5",
    "abst_correta",
    "abst_indevida",
    "alucinacao_sem_resposta",
    "fonte_valida_pct",
    "latencia_mediana_s",
    "latencia_p95_s",
    "perplexidade",
]


def _bertscore_falso(preds, refs, modelo, camadas):
    _bertscore_falso.chamadas += 1
    return [0.8 for _ in preds]


_bertscore_falso.chamadas = 0


def _gold(indices, caminho):
    def evid(trecho):
        return [t.id for t in indices.trechos if trecho in t.texto][:1]

    itens = [
        {
            "id": "q001",
            "pergunta": "Em quanto tempo ocorre a liberação da carga após a atracação?",
            "resposta_ref": "em até 48 horas após a atracação",
            "evidencia": evid("48 horas"),
            "tipo": "factual",
            "tema_id": "2.2",
        },
        {
            "id": "q002",
            "pergunta": "Com quantas horas de antecedência o agente marítimo confirma a chegada do navio?",
            "resposta_ref": "72 horas",
            "evidencia": evid("72 horas"),
            "tipo": "factual",
            "tema_id": "2.1",
        },
        {
            "id": "q003",
            "pergunta": "Qual o salário do supervisor de turno?",
            "resposta_ref": "Não encontrado no manual",
            "evidencia": [],
            "tipo": "sem_resposta",
            "tema_id": "1",
        },
    ]
    escrever_jsonl(caminho, itens)
    return caminho


def test_avaliar_grade_completa_e_retomada(cfg, indices, client_fake, tmp_path):
    gold = _gold(indices, tmp_path / "gold.jsonl")
    configs = [config.carregar_experimento(n) for n in ("S0", "S3")]
    escrever_json(
        "runs/2026-01-01T00-00-00_perplexidade/perplexidade.json", {"perplexidade": {"base": 12.5, "ajustado": 9.75}}
    )
    run_dir, df = avaliar(gold, configs, ["base", "ajustado"], 1, client_fake, bertscore_fn=_bertscore_falso)

    regs = ler_jsonl(run_dir / "respostas.jsonl")
    assert len(regs) == 2 * 2 * 3
    for arq in (
        "config.json",
        "log.txt",
        "metrics.csv",
        "pareado.csv",
        "respostas_por_config.png",
        "recall_por_config.png",
    ):
        assert (run_dir / arq).is_file(), arq
    csv = pd.read_csv(run_dir / "metrics.csv")
    assert list(csv.columns) == COLUNAS_5_7 and len(csv) == 4
    s3 = csv[(csv.config == "S3") & (csv.leitor == "base")].iloc[0]
    assert s3["recall_at_5"] == 1.0 and s3["f1_tokens"] > 0.2 and s3["bertscore_f1"] == pytest.approx(0.8)
    assert s3["abst_correta"] == 1.0 and s3["perplexidade"] == 12.5
    assert math.isnan(csv[csv.config == "S0"].iloc[0]["recall_at_5"])
    assert csv[csv.leitor == "ajustado"].iloc[0]["perplexidade"] == 9.75
    pareado = pd.read_csv(run_dir / "pareado.csv")
    assert set(pareado["metrica"]) == {"em", "f1_tokens"} and set(pareado["config"]) == {"S0", "S3"}

    # log sem texto do manual
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "48 horas" not in log and "agente marítimo" not in log

    # retomada: nada é repetido e o leitor não é chamado de novo
    n_chamadas = len(client_fake.chamadas_chat)
    run2, df2 = avaliar(
        gold, configs, ["base", "ajustado"], 1, client_fake, retomar=run_dir.name, bertscore_fn=_bertscore_falso
    )
    assert run2 == run_dir and len(ler_jsonl(run_dir / "respostas.jsonl")) == 12
    assert len(client_fake.chamadas_chat) == n_chamadas
    assert df2.round(6).equals(df.round(6))


def test_retomada_parcial(cfg, indices, tmp_path):
    gold = _gold(indices, tmp_path / "gold.jsonl")
    configs = [config.carregar_experimento("S1")]
    client = OllamaFake(resposta=leitor_extrativo)
    run_dir, _ = avaliar(gold, configs, ["base"], 2, client, limite=2, bertscore_fn=_bertscore_falso)
    regs = ler_jsonl(run_dir / "respostas.jsonl")
    assert len(regs) == 4 and {r["repeticao"] for r in regs} == {1, 2}
    # simula interrupção: apaga o último registro e retoma
    linhas = (run_dir / "respostas.jsonl").read_text(encoding="utf-8").splitlines()[:-1]
    (run_dir / "respostas.jsonl").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    avaliar(gold, configs, ["base"], 2, client, retomar=run_dir.name, limite=2, bertscore_fn=_bertscore_falso)
    regs = ler_jsonl(run_dir / "respostas.jsonl")
    assert len(regs) == 4 and len({(r["q_id"], r["repeticao"]) for r in regs}) == 4


def test_erros_de_entrada(cfg, indices, client_fake, tmp_path):
    gold = _gold(indices, tmp_path / "gold.jsonl")
    configs = [config.carregar_experimento("S3")]
    with pytest.raises(ErroUsuario, match="não encontrada"):
        avaliar(gold, configs, ["base"], 1, client_fake, retomar="nao-existe")
    sem_modelo = OllamaFake(resposta=leitor_extrativo, modelos=["bge-m3:latest"])
    with pytest.raises(ModeloAusente, match="ollama pull qwen3:4b"):
        avaliar(gold, configs, ["base"], 1, sem_modelo)
    with pytest.raises(ErroUsuario):
        localizar_perplexidade(tmp_path, tmp_path / "nao-existe.json")


def test_cache_do_bertscore(tmp_path):
    chamadas = []

    def fn(preds, refs, modelo, camadas):
        chamadas.append(len(preds))
        return [0.5] * len(preds)

    cache = tmp_path / "cache.json"
    assert bertscore_com_cache(["segredo", "b", "segredo"], ["x", "y", "x"], "m", 9, cache, fn) == [0.5] * 3
    assert chamadas == [2]  # pares repetidos calculados uma vez
    bertscore_com_cache(["segredo", "c"], ["x", "z"], "m", 9, cache, fn)
    assert chamadas == [2, 1]
    assert "segredo" not in cache.read_text()  # o cache guarda só hashes
