"""Cliente mínimo para a API local do Ollama (/api/chat e /api/embed).

Por exigência de confidencialidade, só aceita servidores locais (localhost, 127.0.0.1, ::1).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

from .utils import ErroUsuario

URL_PADRAO = "http://localhost:11434"
HOSTS_LOCAIS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class RespostaChat:
    texto: str
    modelo: str
    latencia_s: float
    tokens_prompt: int | None = None
    tokens_resposta: int | None = None
    bruto: dict = field(default_factory=dict, repr=False)


def validar_url_local(url: str) -> str:
    host = urlparse(url).hostname or ""
    if host not in HOSTS_LOCAIS:
        raise ErroUsuario(
            f"URL do Ollama não é local ({url}). O manual é confidencial e não pode sair da máquina."
        )
    return url.rstrip("/")


class OllamaClient:
    def __init__(self, url: str = URL_PADRAO, timeout: float = 600.0):
        self.url = validar_url_local(url)
        self.timeout = timeout
        self._sem_think: set[str] = set()

    def _post(self, rota: str, payload: dict) -> dict:
        try:
            r = requests.post(f"{self.url}{rota}", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise ErroUsuario(
                f"Não foi possível conectar ao Ollama em {self.url}. Inicie-o com `ollama serve`."
            ) from e
        if r.status_code == 404 and "not found" in r.text.lower():
            modelo = payload.get("model", "?")
            raise ErroUsuario(f"Modelo '{modelo}' não encontrado no Ollama. Rode `ollama pull {modelo}`.")
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama {rota} retornou {r.status_code}: {r.text[:300]}")
        return r.json()

    def listar_modelos(self) -> list[str]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ErroUsuario(
                f"Não foi possível conectar ao Ollama em {self.url}. Inicie-o com `ollama serve`."
            ) from e
        return [m["name"] for m in r.json().get("models", [])]

    def versao(self) -> str:
        try:
            return requests.get(f"{self.url}/api/version", timeout=10).json().get("version", "?")
        except requests.RequestException:
            return "?"

    def chat(
        self,
        modelo: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 8192,
        num_predict: int | None = 200,
        think: bool | None = None,
        formato: str | dict | None = None,
        sistema: str | None = None,
    ) -> RespostaChat:
        mensagens = []
        if sistema:
            mensagens.append({"role": "system", "content": sistema})
        mensagens.append({"role": "user", "content": prompt})
        opcoes = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        if num_predict is not None:
            opcoes["num_predict"] = num_predict
        payload: dict = {"model": modelo, "messages": mensagens, "stream": False, "options": opcoes}
        if formato is not None:
            payload["format"] = formato
        if think is not None and modelo not in self._sem_think:
            payload["think"] = think

        inicio = time.perf_counter()
        try:
            dados = self._post("/api/chat", payload)
        except RuntimeError as e:
            # Modelos sem suporte a "thinking" (ex.: llama3.1) rejeitam o parâmetro.
            if "think" in payload and "think" in str(e).lower():
                self._sem_think.add(modelo)
                payload.pop("think")
                dados = self._post("/api/chat", payload)
            else:
                raise
        latencia = time.perf_counter() - inicio

        texto = _THINK.sub("", dados.get("message", {}).get("content", "")).strip()
        return RespostaChat(
            texto=texto,
            modelo=dados.get("model", modelo),
            latencia_s=latencia,
            tokens_prompt=dados.get("prompt_eval_count"),
            tokens_resposta=dados.get("eval_count"),
            bruto={k: v for k, v in dados.items() if k != "message"},
        )

    def embed(self, modelo: str, textos: list[str]) -> list[list[float]]:
        dados = self._post("/api/embed", {"model": modelo, "input": textos})
        vetores = dados.get("embeddings")
        if not vetores or len(vetores) != len(textos):
            raise RuntimeError(f"Resposta inesperada de /api/embed para {len(textos)} textos.")
        return vetores
