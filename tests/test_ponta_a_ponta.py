"""Fluxo completo da CLI com o manual de exemplo e o Ollama falso."""

import json
from pathlib import Path

import pymupdf

from src.cli import main
from src.utils import escrever_jsonl, ler_json, ler_jsonl

INDEXAR = ["indexar", "--manual", "data/manual.pdf", "--tamanho", "120", "--sobreposicao", "20"]

# (pergunta, resposta de referência, trecho que identifica a passagem-evidência, tipo)
FATOS = [
    ("Qual o prazo para liberação da carga após a atracação?", "até 48 horas após a atracação",
     "48 horas após a atracação", "factual"),
    ("Qual a velocidade máxima de veículos no pátio?", "20 km/h", "velocidade máxima de veículos", "factual"),
    ("Com quantas horas de antecedência o agente marítimo deve confirmar a chegada do navio?", "72 horas",
     "antecedência mínima de 72 horas", "factual"),
    ("Qual a altura máxima de empilhamento de contêineres cheios?", "cinco contêineres cheios",
     "altura máxima de empilhamento", "factual"),
    ("O que o colaborador deve fazer em caso de emergência?",
     "acionar o alarme mais próximo e comunicar a Central de Segurança pelo ramal 190",
     "acionar o alarme mais próximo", "procedimental"),
    ("Como o transportador deve proceder para retirar a carga?",
     "agendar a janela de retirada com antecedência mínima de 24 horas", "agendar a janela de retirada",
     "procedimental"),
    ("A cada quantas horas os portêineres passam por manutenção preventiva?", "a cada 500 horas de operação",
     "500 horas de operação", "factual"),
    ("Qual o valor do salário dos operadores de portêiner?", "Não encontrado no manual", None, "sem_resposta"),
    ("Quem é o presidente da empresa que administra o terminal?", "Não encontrado no manual", None, "sem_resposta"),
]


def _gold(passagens):
    itens = []
    for n, (q, ref, trecho, tipo) in enumerate(FATOS, 1):
        evid = [p["id"] for p in passagens if trecho and trecho in p["texto"]][:1]
        assert tipo == "sem_resposta" or evid, trecho
        itens.append({"id": f"q{n:03d}", "pergunta": q, "resposta_ref": ref, "evidencia": evid, "tipo": tipo,
                      "split": "test"})
    return itens


def _ultimo_run(sufixo: str) -> Path:
    return sorted(Path("runs").glob(f"*_{sufixo}"))[-1]


def test_fluxo_completo(projeto, capsys):
    # Etapas 1-2
    assert main(INDEXAR) == 0
    for arq in ("passages.jsonl", "stats.json", "bm25", "dense.faiss", "dense_ids.json", "manifest.json"):
        assert (projeto / "index" / arq).exists(), arq
    stats = ler_json("index/stats.json")
    assert stats["acima_do_limite"] == 0 and stats["passagens_vazias"] == 0
    passagens = ler_jsonl("index/passages.jsonl")
    assert all(p["secao"] and p["pagina"] >= 1 and p["texto"] for p in passagens)
    assert not any("Página" in p["texto"] or "Manual de Operações" in p["texto"] for p in passagens)
    assert main(["inspecionar", "--n", "3"]) == 0

    # Critério da etapa 2: consultas manuais no top-5 dos dois índices
    gold = _gold(passagens)
    consultas = [{"pergunta": g["pergunta"], "esperada": g["evidencia"]} for g in gold if g["evidencia"]][:5]
    escrever_jsonl("data/consultas.jsonl", consultas)
    capsys.readouterr()
    assert main(["testar-indice", "--consultas", "data/consultas.jsonl"]) == 0
    saida = capsys.readouterr().out
    assert "bm25: 5/5" in saida and "denso: 5/5" in saida

    # Etapa 3
    assert main(["recuperar", "--pergunta", "Qual o prazo para liberação da carga?", "--modo", "hibrido"]) == 0
    assert "4.1 Prazos" in capsys.readouterr().out

    # Etapa 4
    assert main(["perguntar", "--config", "S3", "--pergunta", "Qual o prazo para liberação da carga?"]) == 0
    reg = ler_jsonl(_ultimo_run("perguntar_S3") / "respostas.jsonl")[0]
    assert "48 horas" in reg["resposta"] and not reg["absteve"] and reg["pagina_citada"] == 2
    assert reg["recuperadas"] and reg["prompt"] and reg["latencia_s"] >= 0 and reg["tokens_prompt"]
    Path("data/perguntas.txt").write_text("Qual a velocidade máxima no pátio?\nQuem fundou o terminal?\n")
    assert main(["perguntar", "--config", "S0", "--arquivo", "data/perguntas.txt"]) == 0
    regs = ler_jsonl(_ultimo_run("perguntar_S0") / "respostas.jsonl")
    assert len(regs) == 2 and all(r["recuperadas"] == [] and r["absteve"] for r in regs)

    # Etapa 5
    assert main(["gerar-perguntas", "--n", "10"]) == 0
    cand = ler_jsonl("data/gold_candidatas.jsonl")
    assert len(cand) == 10 and sum(c["tipo"] == "sem_resposta" for c in cand) == 2
    assert {"factual", "procedimental"} <= {c["tipo"] for c in cand}
    assert main(["gerar-perguntas", "--n", "10"]) == 2  # não sobrescreve sem confirmação
    rev_a = [{"id": c["id"], "anotador": "ana", "decisao": "aprovar", "pergunta": c["pergunta"],
              "resposta_ref": c["resposta_ref"], "tipo": c["tipo"], "evidencia": c["evidencia"], "obs": ""} for c in cand]
    rev_b = [dict(r, anotador="beto", decisao="descartar" if i == 0 else "aprovar") for i, r in enumerate(rev_a[:4])]
    escrever_jsonl("data/revisoes/ana.jsonl", rev_a)
    escrever_jsonl("data/revisoes/beto.jsonl", rev_b)
    assert main(["concordancia", "--revisoes", "data/revisoes/ana.jsonl", "data/revisoes/beto.jsonl"]) == 0
    conc = ler_json("data/revisoes/concordancia.json")
    assert conc["itens_comuns"] == 4 and conc["decisao_manter"]["concordancia"] == 0.75
    assert main(["dividir-gold", "--candidatas", "data/gold_candidatas.jsonl",
                 "--revisoes", "data/revisoes/ana.jsonl"]) == 0
    dev, teste = ler_jsonl("data/gold_dev.jsonl"), ler_jsonl("data/gold_test.jsonl")
    assert len(dev) + len(teste) == 10
    secao = {p["id"]: p["secao"] for p in passagens}
    sec = lambda l: {secao[e] for g in l for e in g["evidencia"]}  # noqa: E731
    assert not sec(dev) & sec(teste)

    # Etapa 6 (gold escrito à mão, com fatos conhecidos)
    escrever_jsonl("data/gold_test.jsonl", gold)
    assert main(["avaliar", "--gold", "data/gold_test.jsonl", "--configs", "S0", "S1", "S2", "S3",
                 "--repeticoes", "2"]) == 0
    run = _ultimo_run("avaliar")
    metrics = ler_json(run / "metrics.json")
    assert set(metrics) >= {"S0", "S1", "S2", "S3", "_meta"}
    assert metrics["S0"]["mrr"] is None and metrics["S0"]["abst_correta"] == 1.0
    for cfg in ("S1", "S2", "S3"):
        assert metrics[cfg]["recall@5"] >= 0.8, cfg
        assert metrics[cfg]["reprodutibilidade_respostas"] == 1.0
        assert metrics[cfg]["desvio"]["f1"] == 0.0 and metrics[cfg]["repeticoes"] == 2
    assert metrics["S3"]["f1"] > metrics["S0"]["f1"]
    assert metrics["S3"]["juiz_media"] is not None and metrics["S3"]["fidelidade"] == 1.0
    n_regs = len(ler_jsonl(run / "respostas.jsonl"))
    assert n_regs == len(gold) * 4 * 2
    exemplo = ler_jsonl(run / "respostas.jsonl")[0]
    for campo in ("run_id", "config", "modelo", "q_id", "pergunta", "recuperadas", "resposta", "absteve", "latencia_s"):
        assert campo in exemplo

    # Retomada não repete itens
    assert main(["avaliar", "--gold", "data/gold_test.jsonl", "--configs", "S0", "S1", "S2", "S3",
                 "--repeticoes", "2", "--retomar", str(run)]) == 0
    assert len(ler_jsonl(run / "respostas.jsonl")) == n_regs

    # Validação do juiz (notas manuais simuladas)
    julgaveis = [r for r in ler_jsonl(run / "respostas.jsonl")
                 if r.get("nota_juiz") is not None and r["tipo"] != "sem_resposta" and not r["absteve"]]
    escrever_jsonl(run / "juiz_manual.jsonl", [
        {"config": r["config"], "rep": r["rep"], "q_id": r["q_id"], "nota_manual": r["nota_juiz"],
         "nota_juiz": r["nota_juiz"]} for r in julgaveis[:5]])
    assert main(["validar-juiz", "--run", str(run), "--n", "5"]) == 0
    assert ler_json(run / "metrics.json")["_validacao_juiz"]["concordancia_exata"] == 1.0

    # Etapa 7
    assert main(["analisar", "--run", str(run), "--ks", "1", "3", "5"]) == 0
    for arq in ("analise.md", "comparacoes.json", "recall_k.csv", "recall_k.png", "erros_amostra.jsonl"):
        assert (run / arq).exists(), arq
    comps = json.loads((run / "comparacoes.json").read_text())
    assert len(comps) == 6 and all(0 <= c["mcnemar_p"] <= 1 for c in comps)

    assert main(["varrer", "--gold", "data/gold_test.jsonl", "--manual", "data/manual.pdf",
                 "--tamanhos", "100", "120", "--ks", "2", "3"]) == 2  # nome "test" pede confirmação
    escrever_jsonl("data/gold_dev.jsonl", gold)
    assert main(["varrer", "--gold", "data/gold_dev.jsonl", "--manual", "data/manual.pdf",
                 "--tamanhos", "100", "120", "--ks", "2", "3"]) == 0
    varr = ler_json(_ultimo_run("varredura") / "varredura.json")
    assert len(varr["resultados"]) == 4 and varr["melhor"]["k"] in (2, 3)
    assert (projeto / "index" / "t100" / "manifest.json").exists()
    # o índice base usa sobreposição 20 (≠ padrão 50), então t120 também precisa ser construído
    assert (projeto / "index" / "t120" / "manifest.json").exists()


def test_erros_de_entrada(projeto, capsys):
    assert main(["indexar"]) == 2                                  # argumento ausente sem terminal
    assert main(["indexar", "--manual", "data/nao_existe.pdf"]) == 2
    Path("data/manual.txt").write_text("x")
    assert main(["indexar", "--manual", "data/manual.txt"]) == 2   # extensão errada
    assert main(["perguntar", "--config", "S3", "--pergunta", "x"]) == 2  # índice ausente
    assert "indexar" in capsys.readouterr().err
    assert main(["perguntar", "--config", "S9", "--pergunta", "x"]) == 2

    # PDF sem camada de texto
    doc = pymupdf.open()
    doc.new_page()
    doc.save("data/digitalizado.pdf")
    assert main(["indexar", "--manual", "data/digitalizado.pdf"]) == 2
    assert "ocrmypdf" in capsys.readouterr().err

    # Manual alterado após a indexação → índice desatualizado
    assert main(INDEXAR) == 0
    doc = pymupdf.open("data/manual.pdf")
    doc[0].insert_text((100, 100), "alteração")
    doc.saveIncr()
    doc.close()
    capsys.readouterr()
    assert main(["perguntar", "--config", "S3", "--pergunta", "Qual o prazo?"]) == 2
    assert "mudou desde a indexação" in capsys.readouterr().err
