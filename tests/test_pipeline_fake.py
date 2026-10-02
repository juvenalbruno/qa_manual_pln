from __future__ import annotations

import dataclasses
import json

from qa_manual import config
from qa_manual.generate import formatar_trechos, montar_prompt, remover_think
from qa_manual.io_utils import anexar_log_arquivo, remover_log_arquivo
from qa_manual.ollama_client import OllamaFake
from qa_manual.pipeline import perguntar

CAMPOS_5_6 = {
    "run_id",
    "config",
    "leitor",
    "modelo",
    "q_id",
    "pergunta",
    "tipo",
    "recuperadas",
    "resposta",
    "absteve",
    "citou_fonte",
    "fonte_valida",
    "citacoes",
    "latencia_s",
    "t_embedding_s",
    "t_busca_s",
    "t_leitura_s",
    "n_tokens_prompt",
    "n_tokens_resposta",
}
PERGUNTA = "Em quanto tempo ocorre a liberação da carga após a atracação?"


def test_s3_registro_completo(indices, client_fake):
    s3 = config.carregar_experimento("S3")
    reg = perguntar(PERGUNTA, s3, "base", indices, client_fake, q_id="q001", tipo="factual", run_id="r1")
    assert CAMPOS_5_6 <= set(reg)
    assert reg["run_id"] == "r1" and reg["config"] == "S3" and reg["modelo"] == "qwen3:4b"
    assert "48 horas" in reg["resposta"] and not reg["absteve"]
    assert reg["citou_fonte"] and reg["fonte_valida"] and reg["citacoes"][0]["tema_id"] == "2.2"
    assert 0 < len(reg["recuperadas"]) <= s3.k and set(reg["recuperadas"][0]) == {"id", "score", "rank"}
    assert reg["n_tokens_prompt"] > 0 and reg["latencia_s"] >= reg["t_leitura_s"]
    chamada = client_fake.chamadas_chat[-1]
    assert chamada["options"] == {"temperature": 0.0, "top_p": 1.0, "seed": 42, "num_ctx": 8192, "num_predict": 200}
    assert chamada["think"] is False and "Trechos:" in chamada["prompt"]
    json.dumps(reg)  # serializável


def test_s0_sem_trechos(cfg, client_fake):
    s0 = config.carregar_experimento("S0")
    reg = perguntar(PERGUNTA, s0, "ajustado", None, client_fake, q_id="q002", tipo="factual")
    assert CAMPOS_5_6 <= set(reg)
    assert reg["recuperadas"] == [] and reg["modelo"] == "qwen3-manual:4b"
    assert reg["t_embedding_s"] == 0.0 and reg["t_busca_s"] == 0.0
    assert not reg["fonte_valida"] and reg["absteve"]  # o leitor falso se abstém sem trechos
    assert "Trechos:" not in client_fake.chamadas_chat[-1]["prompt"]


def test_limiar_de_score_abstem_sem_chamar_o_llm(cfg, indices, client_fake):
    s3 = config.carregar_experimento("S3")
    s3 = s3.com(abstencao=dataclasses.replace(s3.abstencao, limiar_score=10.0))
    reg = perguntar(PERGUNTA, s3, "base", indices, client_fake)
    assert reg["absteve"] and reg["resposta"] == "Não encontrado no manual." and reg["t_leitura_s"] == 0.0
    assert client_fake.chamadas_chat == []


def test_bloco_think_removido(cfg, indices, caplog):
    client = OllamaFake(resposta="<think>raciocínio interno</think>Em até 48 horas (seção 2.2, p. 5).")
    reg = perguntar(PERGUNTA, config.carregar_experimento("S1"), "base", indices, client)
    assert reg["resposta"] == "Em até 48 horas (seção 2.2, p. 5)."
    assert "think" in caplog.text
    assert remover_think("<think>sem fim") == ("", True)
    assert remover_think("antes</think>depois") == ("depois", True)
    assert remover_think("normal") == ("normal", False)


def test_formato_do_prompt(indices):
    trechos = indices.trechos[:2]
    texto = formatar_trechos(trechos)
    assert texto.startswith(f"[1] (seção {trechos[0].tema_id}, p. {trechos[0].pagina_inicio}) ")
    prompt = montar_prompt("Pergunta de teste?", trechos)
    assert "(seção {tema_id}, p. {pagina})" in prompt and prompt.rstrip().endswith("Resposta:")
    assert "Pergunta: Pergunta de teste?" in prompt


def test_log_nao_contem_texto_de_trecho(cfg, indices, client_fake, tmp_path):
    arq = tmp_path / "log.txt"
    handler = anexar_log_arquivo(arq)
    try:
        for exp in ("S0", "S1", "S2", "S3"):
            perguntar(PERGUNTA, config.carregar_experimento(exp), "base", indices, client_fake, q_id="q9")
    finally:
        remover_log_arquivo(handler)
    log = arq.read_text(encoding="utf-8")
    assert "q=q9" in log
    for t in indices.trechos:
        palavras = t.texto.split()
        janelas = {" ".join(palavras[i : i + 5]) for i in range(0, max(1, len(palavras) - 4))}
        assert not any(j in log for j in janelas), t.id
    assert PERGUNTA not in log
