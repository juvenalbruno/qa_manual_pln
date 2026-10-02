"""OllamaReal sem servidor: conexão recusada e mapeamento de erros com um cliente substituto."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import ollama
import pytest

from qa_manual.ollama_client import ModeloAusente, OllamaIndisponivel, OllamaReal


def _porta_livre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ollama_fora_do_ar():
    client = OllamaReal(f"http://127.0.0.1:{_porta_livre()}", timeout_s=2)
    with pytest.raises(OllamaIndisponivel, match="ollama serve"):
        client.modelos_disponiveis()
    assert client.versao() == "?"


class _Substituto:
    def __init__(self, erros=()):
        self.erros = list(erros)
        self.chamadas = []

    def chat(self, **kwargs):
        self.chamadas.append(kwargs)
        if self.erros:
            raise self.erros.pop(0)
        return SimpleNamespace(
            message=SimpleNamespace(content="ok"), model=kwargs["model"], prompt_eval_count=7, eval_count=1
        )

    def embed(self, model, input):
        return SimpleNamespace(embeddings=[[0.1, 0.2] for _ in input])

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="qwen3:4b"), SimpleNamespace(model=None)])


def _cliente(substituto):
    c = OllamaReal("http://localhost:11434")
    c._cliente = substituto
    return c


def test_chat_embed_e_lista():
    sub = _Substituto()
    c = _cliente(sub)
    r = c.chat("qwen3:4b", [{"role": "user", "content": "oi"}], {"temperature": 0}, think=False, format="json")
    assert r == {"content": "ok", "model": "qwen3:4b", "prompt_eval_count": 7, "eval_count": 1}
    assert sub.chamadas[0]["think"] is False and sub.chamadas[0]["format"] == "json"
    assert c.embed("bge-m3", ["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert c.modelos_disponiveis() == ["qwen3:4b"]


def test_modelo_sem_suporte_a_think_e_modelo_ausente():
    sub = _Substituto([ollama.ResponseError('"llama3.2:3b" does not support thinking', 400)])
    c = _cliente(sub)
    assert c.chat("llama3.2:3b", [{"role": "user", "content": "oi"}], {}, think=False)["content"] == "ok"
    assert "think" in sub.chamadas[0] and "think" not in sub.chamadas[1]
    c.chat("llama3.2:3b", [{"role": "user", "content": "oi"}], {}, think=False)
    assert "think" not in sub.chamadas[2]  # lembra que o modelo não aceita think

    c = _cliente(_Substituto([ollama.ResponseError("model 'qwen3-manual:4b' not found", 404)]))
    with pytest.raises(ModeloAusente, match="ollama create qwen3-manual:4b"):
        c.chat("qwen3-manual:4b", [], {})
    c = _cliente(_Substituto([ollama.ResponseError("erro interno", 500)]))
    with pytest.raises(ollama.ResponseError):
        c.chat("qwen3:4b", [], {})
