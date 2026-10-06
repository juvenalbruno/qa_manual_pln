from __future__ import annotations

import json

import pytest

from qa_manual import config, prompts
from qa_manual.io_utils import ErroUsuario
from qa_manual.ollama_client import instrucao_modelo, modelo_presente, validar_host_local


def test_base_e_experimentos(cfg):
    assert cfg.nome == "base" and cfg.retriever == "hibrido" and cfg.k == 5
    esperado = {"S0": "nenhum", "S1": "bm25", "S2": "denso", "S3": "hibrido"}
    for nome, modo in esperado.items():
        c = config.carregar_experimento(nome)
        assert c.nome == nome and c.retriever == modo
        assert c.decodificacao == cfg.decodificacao and c.k == cfg.k  # só o recuperador muda
    assert cfg.modelo_leitor("base") == "qwen3:4b" and cfg.modelo_leitor("ajustado") == "qwen3-manual:4b"
    with pytest.raises(ErroUsuario):
        cfg.modelo_leitor("outro")
    json.dumps(cfg.como_dict())


def _yaml(tmp_path, nome, conteudo):
    p = tmp_path / nome
    p.write_text(conteudo, encoding="utf-8")
    return p


def test_sobrescrita_parcial_e_chaves_invalidas(cfg, tmp_path):
    c = config.carregar("configs/base.yaml", _yaml(tmp_path, "x.yaml", "trechos:\n  max_tokens: 300\nk: 8\n"))
    assert c.trechos.max_tokens == 300 and c.trechos.min_tokens == 60 and c.k == 8 and c.nome == "x"
    with pytest.raises(ErroUsuario, match="desconhecida"):
        config.carregar("configs/base.yaml", _yaml(tmp_path, "a.yaml", "nao_existe: 1\n"))
    with pytest.raises(ErroUsuario, match="desconhecida"):
        config.carregar("configs/base.yaml", _yaml(tmp_path, "b.yaml", "bm25:\n  k3: 1\n"))
    with pytest.raises(ErroUsuario, match="inteiro"):
        config.carregar("configs/base.yaml", _yaml(tmp_path, "c.yaml", "k: cinco\n"))
    with pytest.raises(ErroUsuario, match="retriever"):
        config.carregar("configs/base.yaml", _yaml(tmp_path, "d.yaml", "retriever: magico\n"))
    with pytest.raises(ErroUsuario, match="não encontrada"):
        config.carregar("configs/base.yaml", "S9")
    with pytest.raises(ErroUsuario, match="YAML"):
        config.carregar("configs/base.yaml", _yaml(tmp_path, "e.yaml", "k: [\n"))
    c = config.carregar("configs/base.yaml", _yaml(tmp_path, "f.yaml", "abstencao:\n  limiar_score: 0.02\n"))
    assert c.abstencao.limiar_score == 0.02


def test_ollama_host_do_ambiente(cfg, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    assert config.carregar().ollama.host == "http://127.0.0.1:11500"


def test_host_local_e_modelos():
    assert validar_host_local("http://localhost:11434/") == "http://localhost:11434"
    assert validar_host_local("127.0.0.1:11434") == "http://127.0.0.1:11434"
    with pytest.raises(ErroUsuario, match="não é local"):
        validar_host_local("https://api.exemplo.com")
    assert modelo_presente("bge-m3", ["bge-m3:latest"]) and modelo_presente("qwen3:4b", ["qwen3:4b"])
    assert not modelo_presente("qwen3:8b", ["qwen3:4b"])
    assert "ollama pull qwen3:4b" in instrucao_modelo("qwen3:4b")
    assert "ollama create qwen3-manual:4b" in instrucao_modelo("qwen3-manual:4b")


def test_prompts_existem_com_placeholders():
    for nome, esperados in prompts.PLACEHOLDERS.items():
        assert prompts.caminho(nome).is_file(), nome
        assert prompts.placeholders(prompts.carregar(nome)) == esperados, nome
    texto = prompts.preencher(prompts.carregar("gerador_perguntas"), tipo="factual", texto="T {x}")
    assert '{"pergunta": "...", "resposta": "..."}' in texto and "T {x}" in texto
    with pytest.raises(ErroUsuario):
        prompts.preencher(prompts.carregar("leitor"), pergunta="só a pergunta")
    with pytest.raises(ErroUsuario):
        prompts.carregar("inexistente")
