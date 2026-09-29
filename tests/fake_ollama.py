"""Servidor HTTP que imita a API do Ollama, para testar o pipeline sem modelos reais.

- /api/embed: vetores de "hashing trick" sobre os tokens do BM25 (determinísticos e com semântica léxica).
- /api/chat: leitor extrativo (frase da passagem com maior sobreposição com a pergunta), juiz por F1,
  gerador de perguntas em JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from src.evaluate import f1_tokens
from src.utils import tokenizar_bm25

DIM = 256
_PASSAGEM = re.compile(r"^\[(\d+)\] \(seção (.+?), p\. (\d+)\) (.*)$", re.MULTILINE)


def vetor(texto: str) -> list[float]:
    v = np.zeros(DIM, dtype="float32")
    for tok in tokenizar_bm25(texto):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    return v.tolist()


def _campo(prompt: str, rotulo: str) -> str:
    m = re.search(rf"^{rotulo}:\s*(.*)$", prompt, re.MULTILINE)
    return m.group(1).strip() if m else ""


def responder_chat(prompt: str) -> str:
    if "Tipo de pergunta desejado" in prompt:
        trecho = prompt.split("Trecho (", 1)[1].split("): ", 1)[1]
        frase = re.split(r"(?<=[.!?])\s+", trecho.strip())[0]
        palavras = frase.split()
        return json.dumps({"pergunta": f"O que o manual diz sobre {' '.join(palavras[1:7])}?",
                           "resposta": " ".join(palavras[:12])})
    if "NÃO esteja no manual" in prompt:
        n = int(re.search(r"Escreva (\d+) perguntas", prompt).group(1))
        return json.dumps({"perguntas": [f"Qual o preço do serviço número {i} do terminal?" for i in range(n)]})
    if prompt.startswith("Compare a resposta"):
        f1 = f1_tokens(_campo(prompt, "Resposta do sistema"), _campo(prompt, "Referência"))
        return "2" if f1 >= 0.5 else "1" if f1 >= 0.2 else "0"
    if prompt.startswith("Verifique se a resposta"):
        return "1"
    pergunta = _campo(prompt, "Pergunta")
    passagens = _PASSAGEM.findall(prompt)
    if not passagens:
        return "Não sei."
    q = set(tokenizar_bm25(pergunta))
    melhor, melhor_score = None, 0
    for _, secao, pagina, texto in passagens:
        for frase in re.split(r"(?<=[.!?])\s+", texto):
            score = len(q & set(tokenizar_bm25(frase)))
            if score > melhor_score:
                melhor, melhor_score = (frase.rstrip("."), secao, pagina), score
    if melhor is None or melhor_score < 2:
        return "Não encontrado no manual."
    return f"{melhor[0]} (seção {melhor[1]}, p. {melhor[2]})."


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silencioso
        pass

    def _json(self, dados: dict, status: int = 200) -> None:
        corpo = json.dumps(dados).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": m} for m in ("qwen3:8b", "bge-m3:latest", "llama3.1:8b")]})
        elif self.path == "/api/version":
            self._json({"version": "fake"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        dados = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            entradas = dados["input"] if isinstance(dados["input"], list) else [dados["input"]]
            self._json({"model": dados["model"], "embeddings": [vetor(t) for t in entradas]})
        elif self.path == "/api/chat":
            if "think" in dados and dados["model"].startswith("llama"):
                self._json({"error": f"\"{dados['model']}\" does not support thinking"}, 400)
                return
            prompt = dados["messages"][-1]["content"]
            texto = responder_chat(prompt)
            self._json({
                "model": dados["model"], "message": {"role": "assistant", "content": texto},
                "prompt_eval_count": len(prompt.split()), "eval_count": len(texto.split()), "done": True,
            })
        else:
            self._json({"error": "not found"}, 404)


class FakeOllama:
    def __init__(self):
        self.servidor = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.servidor.server_address[1]}"
        self.thread = threading.Thread(target=self.servidor.serve_forever, daemon=True)

    def __enter__(self) -> "FakeOllama":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.servidor.shutdown()
