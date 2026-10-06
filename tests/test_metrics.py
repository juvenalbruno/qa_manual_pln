from __future__ import annotations

import math

import pytest

from qa_manual.metrics import (
    agregar,
    bootstrap_ic,
    exact_match,
    f1_tokens,
    ganho_pareado,
    mrr_at_k,
    normalizar,
    pontuar,
    recall_at_k,
)


def test_normalizar():
    assert normalizar("Até 48 horas, após a atracação!") == "até 48 horas após atracação"
    assert normalizar("  O   prazo  do navio ") == "prazo navio"
    assert normalizar("1.420 m") == normalizar("1420 m")
    assert normalizar("o de a") == ""


def test_exact_match_e_f1():
    assert exact_match("Até 48 horas após a atracação.", "até 48 horas após atracação") == 1
    assert exact_match("48 horas", "72 horas") == 0
    # referência normalizada "até 48 horas após atracação" (5 tokens); predição "48 horas": P = 1, R = 2/5
    assert f1_tokens("48 horas", "até 48 horas após a atracação") == pytest.approx(2 * 0.4 / 1.4)
    assert f1_tokens("", "algo") == 0.0
    assert f1_tokens("", "") == 1.0
    assert f1_tokens("o de a", "da do") == 1.0  # só artigos/preposições: ambas vazias
    assert f1_tokens("liberação", "liberacao") == 0.0  # acentos mantidos
    assert f1_tokens("abc", "xyz") == 0.0


def test_recall_e_mrr():
    assert recall_at_k(["t1", "t2"], ["t2"]) == 1 and recall_at_k(["t1"], ["t9"]) == 0
    assert mrr_at_k(["t1", "t2", "t3"], ["t3", "t2"]) == 0.5
    assert mrr_at_k(["t1"], ["t9"]) == 0.0


def _reg(config, leitor, q_id, resposta, absteve, recuperadas, retriever="hibrido", fonte=True, lat=2.0):
    return {
        "config": config,
        "leitor": leitor,
        "q_id": q_id,
        "resposta": resposta,
        "absteve": absteve,
        "fonte_valida": fonte,
        "latencia_s": lat,
        "retriever": retriever,
        "recuperadas": [{"id": i} for i in recuperadas],
    }


GOLD = {
    "q1": {"tipo": "factual", "resposta_ref": "48 horas", "evidencia": ["t1"]},
    "q2": {"tipo": "procedimental", "resposta_ref": "agendar a retirada", "evidencia": ["t2"]},
    "q3": {"tipo": "sem_resposta", "resposta_ref": "Não encontrado no manual", "evidencia": []},
}


def test_pontuar_e_agregar_com_sem_resposta():
    regs = [
        _reg("S3", "base", "q1", "48 horas (seção 2.2, p. 5).", False, ["t1", "t5"], lat=1.0),
        _reg("S3", "base", "q2", "Não encontrado no manual.", True, ["t9", "t2"], fonte=False, lat=2.0),
        _reg("S3", "base", "q3", "Não encontrado no manual.", True, ["t4"], fonte=False, lat=3.0),
        _reg("S0", "base", "q1", "48 horas.", False, [], retriever="nenhum", fonte=False),
        _reg("S0", "base", "q3", "Algo inventado.", False, [], retriever="nenhum", fonte=False),
    ]
    pontuados = [pontuar(r, GOLD[r["q_id"]]) for r in regs]
    assert pontuados[0]["em"] == 1 and pontuados[0]["recall"] == 1 and pontuados[0]["mrr"] == 1.0
    assert pontuados[1]["em"] == 0 and pontuados[1]["f1_tokens"] == 0.0 and pontuados[1]["mrr"] == 0.5
    assert math.isnan(pontuados[2]["em"])
    linhas = {(l["config"], l["leitor"]): l for l in agregar(pontuados)}
    s3 = linhas[("S3", "base")]
    assert s3["n"] == 3 and s3["em"] == 0.5 and s3["recall_at_5"] == 1.0 and s3["mrr_at_5"] == 0.75
    assert s3["abst_correta"] == 1.0 and s3["alucinacao_sem_resposta"] == 0.0 and s3["abst_indevida"] == 0.5
    assert s3["fonte_valida_pct"] == 1.0 and s3["latencia_mediana_s"] == 2.0
    assert s3["latencia_p95_s"] == pytest.approx(2.9)
    s0 = linhas[("S0", "base")]
    assert math.isnan(s0["recall_at_5"]) and s0["abst_correta"] == 0.0 and s0["alucinacao_sem_resposta"] == 1.0
    assert s0["fonte_valida_pct"] == 0.0


def test_ganho_pareado_e_bootstrap():
    regs = []
    for q in ("q1", "q2"):
        regs.append(_reg("S3", "base", q, "errado", False, ["t1"]))
        regs.append(_reg("S3", "ajustado", q, GOLD[q]["resposta_ref"], False, ["t1"]))
    pontuados = [pontuar(r, GOLD[r["q_id"]]) for r in regs]
    linhas = ganho_pareado(pontuados, n_reamostras=200)
    em = next(l for l in linhas if l["metrica"] == "em")
    assert em["n_perguntas"] == 2 and em["ganho_medio"] == 1.0 and em["ic95_inf"] == 1.0
    media, inf, sup = bootstrap_ic([0.0, 1.0] * 10, n_reamostras=500)
    assert media == 0.5 and inf < 0.5 < sup
    assert all(math.isnan(v) for v in bootstrap_ic([]))
