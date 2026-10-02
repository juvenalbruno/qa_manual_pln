from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from qa_manual import gold
from qa_manual.chunking import Trecho
from qa_manual.io_utils import ErroUsuario
from qa_manual.ollama_client import OllamaFake

FRASE = "Não encontrado no manual"


def _trechos(n_temas=6, por_tema=4, tokens=200):
    ts = []
    for t in range(1, n_temas + 1):
        for _ in range(por_tema):
            i = len(ts) + 1
            ts.append(
                Trecho(
                    id=f"t{i:04d}",
                    tema=f"{t} Tema {t}",
                    tema_id=str(t),
                    pagina_inicio=t,
                    pagina_fim=t,
                    paragrafos=1,
                    n_tokens=tokens,
                    texto=f"Texto fictício {i} sobre o tema {t}.",
                )
            )
    return ts


def test_amostragem_estratificada():
    ts = _trechos() + [Trecho("t0999", "9 Curto", "9", 9, 9, 1, 50, "curto")]
    amostra = gold.amostrar_trechos(ts, 12, seed=1)
    assert len(amostra) == 12 and len({t.id for t in amostra}) == 12
    assert Counter(t.tema_id for t in amostra) == {str(i): 2 for i in range(1, 7)}
    assert all(t.n_tokens >= gold.MIN_TOKENS_AMOSTRA for t in amostra)
    assert [t.id for t in gold.amostrar_trechos(ts, 12, seed=1)] == [t.id for t in amostra]
    assert len({t.tema_id for t in gold.amostrar_trechos(ts, 6, seed=2)}) == 6  # 1 por tema
    assert len(gold.amostrar_trechos(ts, 100, seed=1)) == 24
    assert gold.amostrar_trechos([], 5, seed=1) == []


def test_extrair_json():
    assert gold.extrair_json('Claro! {"pergunta": "Q?"} Espero ter ajudado.') == {"pergunta": "Q?"}
    assert gold.extrair_json("sem json") is None
    assert gold.extrair_json("{quebrado") is None


def _gerador(modelo, prompt):
    if "NÃO aparece" in prompt:
        return json.dumps({"pergunta": "Qual o valor da taxa fictícia?"})
    tipo = "procedimental" if '"procedimental"' in prompt.split("\n")[2] else "factual"
    return json.dumps({"pergunta": f"Pergunta {tipo}?", "resposta": "Resposta curta."})


def test_gerar_candidatas(cfg):
    client = OllamaFake(resposta=_gerador)
    candidatas, stats = gold.gerar_candidatas(cfg, client, _trechos(), n=20)
    assert len(candidatas) == 20 and stats["json_valido_pct"] == 1.0
    tipos = Counter(c["tipo"] for c in candidatas)
    assert tipos == {"factual": 10, "procedimental": 6, "sem_resposta": 4}
    sem = [c for c in candidatas if c["tipo"] == "sem_resposta"]
    assert all(c["resposta_ref"] == FRASE and c["evidencia"] == [] for c in sem)
    resp = [c for c in candidatas if c["tipo"] != "sem_resposta"]
    assert all(len(c["evidencia"]) == 1 and c["tema_id"] and not c["revisado"] for c in resp)
    assert [c["id"] for c in candidatas] == [f"q{i:03d}" for i in range(1, 21)]
    assert all(ch["format"] == "json" and ch["model"] == "llama3.2:3b" for ch in client.chamadas_chat)


def test_gerador_invalido_tenta_tres_vezes(cfg):
    respostas = iter(["lixo", "{}", json.dumps({"pergunta": "Q?", "resposta": "R."})] + ["lixo"] * 10)
    client = OllamaFake(resposta=lambda m, p: next(respostas))
    candidatas, stats = gold.gerar_candidatas(cfg, client, _trechos(n_temas=2, por_tema=1), n=2)
    assert (
        len(candidatas) == 1 and stats["tentativas"] == 6 and stats["json_valido_pct"] == pytest.approx(1 / 6, abs=1e-3)
    )
    seeds = [c["options"]["seed"] for c in client.chamadas_chat[:3]]
    assert seeds == [42, 43, 44]  # semente muda a cada tentativa


def _revisar(caminho: Path, decisoes: dict[str, tuple]) -> None:
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    for ln in linhas:
        d = decisoes.get(ln["id"], ("aceitar",))
        ln["decisao"] = d[0]
        if d[0] == "editar":
            ln["pergunta_corrigida"], ln["resposta_corrigida"] = d[1], d[2]
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gold.COLUNAS_REVISAO, delimiter=";")
        w.writeheader()
        w.writerows(linhas)


def test_exportar_e_importar_revisao(cfg, tmp_path):
    ts = _trechos()
    candidatas, _ = gold.gerar_candidatas(cfg, OllamaFake(resposta=_gerador), ts, n=10)
    csv_path = tmp_path / "revisao.csv"
    assert gold.exportar_revisao(candidatas, ts, csv_path) == 10
    bruto = csv_path.read_bytes()
    assert bruto.startswith(b"\xef\xbb\xbf") and b";" in bruto.split(b"\n")[0]
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    assert list(linhas[0]) == gold.COLUNAS_REVISAO and linhas[0]["texto_evidencia"]

    with pytest.raises(ErroUsuario, match="sem decisão"):
        gold.importar_revisao(csv_path, FRASE)
    _revisar(csv_path, {"q001": ("editar", "Pergunta corrigida?", ""), "q002": ("descartar",)})
    itens = gold.importar_revisao(csv_path, FRASE)
    assert len(itens) == 9 and all(it["revisado"] for it in itens)
    q1 = next(it for it in itens if it["id"] == "q001")
    assert q1["pergunta"] == "Pergunta corrigida?" and q1["resposta_ref"] == "Resposta curta."
    assert q1["origem"] == "editado" and q1["evidencia"] == candidatas[0]["evidencia"]

    _revisar(csv_path, {"q003": ("editar", "", "")})
    with pytest.raises(ErroUsuario, match="editar"):
        gold.importar_revisao(csv_path, FRASE)
    _revisar(csv_path, {"q003": ("talvez",)})
    with pytest.raises(ErroUsuario, match="inválida"):
        gold.importar_revisao(csv_path, FRASE)


def test_dividir_por_tema_sem_vazamento():
    itens = []
    for t in range(10):
        for j in range(5):
            tipo = "sem_resposta" if j == 4 else "factual"
            itens.append(
                {
                    "id": f"q{len(itens):03d}",
                    "tema_id": str(t),
                    "tipo": tipo,
                    "evidencia": [] if tipo == "sem_resposta" else [f"t{t}"],
                }
            )
    dev, test = gold.dividir(itens, 0.3, seed=7)
    assert len(dev) + len(test) == 50 and 0.2 <= len(dev) / 50 <= 0.4
    assert not {i["tema_id"] for i in dev} & {i["tema_id"] for i in test}
    assert any(i["tipo"] == "sem_resposta" for i in dev) and any(i["tipo"] == "sem_resposta" for i in test)
    assert {i["split"] for i in dev} == {"dev"} and {i["split"] for i in test} == {"test"}
    assert gold.dividir(itens, 0.3, seed=7) == (dev, test)


def test_dividir_redistribui_sem_resposta_quando_preciso():
    itens = [{"id": f"q{i}", "tema_id": str(i % 3), "tipo": "factual", "evidencia": ["t"]} for i in range(9)]
    itens += [
        {"id": "s1", "tema_id": "9", "tipo": "sem_resposta", "evidencia": []},
        {"id": "s2", "tema_id": "9", "tipo": "sem_resposta", "evidencia": []},
    ]
    dev, test = gold.dividir(itens, 0.3, seed=1)
    assert any(i["tipo"] == "sem_resposta" for i in dev) and any(i["tipo"] == "sem_resposta" for i in test)
    with pytest.raises(ErroUsuario):
        gold.dividir([{"id": "a", "tema_id": "1", "tipo": "factual"}], 0.3, 1)
    with pytest.raises(ErroUsuario):
        gold.dividir([], 0.3, 1)


def test_reservar_ppl(tmp_path):
    ts = _trechos()
    gold_test = [{"id": "q1", "tema_id": "1"}, {"id": "q2", "tema_id": "2"}]
    destino = tmp_path / "ppl.txt"
    escolhidos = gold.reservar_ppl(ts, gold_test, n=5, seed=3, destino=destino)
    assert len(escolhidos) == 5 and not {t.tema_id for t in escolhidos} & {"1", "2"}
    assert destino.read_text(encoding="utf-8").count("\n\n") == 4
    assert len(gold.reservar_ppl(ts, None, n=100)) == len(ts)


def test_validar_gold():
    gold.validar_gold([{"id": "q1", "pergunta": "P", "resposta_ref": "R", "evidencia": [], "tipo": "factual"}], "x")
    with pytest.raises(ErroUsuario, match="vazio"):
        gold.validar_gold([], "x")
    with pytest.raises(ErroUsuario, match="ausente"):
        gold.validar_gold([{"id": "q1", "pergunta": "P"}], "x")
    with pytest.raises(ErroUsuario, match="inválido"):
        gold.validar_gold([{"id": "q1", "pergunta": "P", "resposta_ref": "R", "evidencia": [], "tipo": "x"}], "x")
