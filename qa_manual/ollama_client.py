"""Cliente fino sobre o Ollama (embeddings e chat), com uma implementação falsa para testes.

Confidencialidade: :class:`OllamaReal` só aceita servidores locais (localhost, 127.0.0.1, ::1). Nenhum texto do
manual pode ser enviado a outro endereço.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

from .io_utils import ErroUsuario

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)

HOSTS_LOCAIS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
MODELOS_CRIADOS_LOCALMENTE = {"qwen3-manual:4b"}


class OllamaIndisponivel(ErroUsuario):
    """O servidor do Ollama não respondeu."""

    codigo = 1


class ModeloAusente(ErroUsuario):
    """O modelo pedido não está instalado no Ollama."""

    codigo = 1


def instrucao_modelo(modelo: str) -> str:
    """Mensagem que diz como obter um modelo ausente."""
    if modelo in MODELOS_CRIADOS_LOCALMENTE or modelo.endswith("-manual:4b"):
        return (
            f"modelo {modelo} ausente; crie com `ollama create {modelo} -f Modelfile` "
            "(ver docs/README.md, seção Ajuste fino no Colab)"
        )
    return f"modelo {modelo} ausente; rode `ollama pull {modelo}`"


def modelo_presente(modelo: str, disponiveis: list[str]) -> bool:
    """Compara nomes do Ollama tolerando o sufixo implícito ``:latest``."""
    alvo = modelo if ":" in modelo else f"{modelo}:latest"
    return any(d == modelo or d == alvo for d in disponiveis)


def validar_host_local(host: str) -> str:
    """Garante que ``host`` aponta para a própria máquina e devolve a URL sem barra final."""
    if "://" not in host:
        host = f"http://{host}"
    nome = urlparse(host).hostname or ""
    if nome not in HOSTS_LOCAIS:
        raise ErroUsuario(
            f"O endereço do Ollama não é local ({host}). O manual é confidencial e não pode sair da máquina."
        )
    return host.rstrip("/")


class OllamaClient(Protocol):
    """Interface usada pelo restante do código (permite injetar :class:`OllamaFake` nos testes)."""

    def embed(self, model: str, input: list[str]) -> list[list[float]]:
        """Embeddings de uma lista de textos."""
        ...

    def chat(
        self,
        model: str,
        messages: list[dict],
        options: dict,
        think: bool | None = None,
        format: str | None = None,
    ) -> dict:
        """Uma rodada de chat; devolve ``{"content", "model", "prompt_eval_count", "eval_count"}``."""
        ...

    def modelos_disponiveis(self) -> list[str]:
        """Nomes dos modelos instalados."""
        ...


class OllamaReal:
    """Implementação sobre o pacote ``ollama`` (``ollama.Client(host=...)``).

    ``chat`` devolve ``{"content", "model", "prompt_eval_count", "eval_count"}``.
    """

    def __init__(self, host: str = "http://localhost:11434", timeout_s: float = 300):
        import ollama

        self.host = validar_host_local(host)
        self.timeout_s = timeout_s
        self._ollama = ollama
        self._cliente = ollama.Client(host=self.host, timeout=timeout_s)
        self._sem_think: set[str] = set()

    def _chamar(self, fn: Callable, modelo: str | None = None):
        import httpx

        try:
            return fn()
        except self._ollama.ResponseError as e:
            if e.status_code == 404 and modelo:
                raise ModeloAusente(instrucao_modelo(modelo)) from None
            raise
        except (ConnectionError, httpx.ConnectError):
            raise OllamaIndisponivel(f"Ollama não respondeu em {self.host}; rode `ollama serve`") from None
        except httpx.TimeoutException:
            raise OllamaIndisponivel(
                f"Ollama em {self.host} não respondeu em {self.timeout_s:.0f} s; aumente ollama.timeout_s"
            ) from None

    def embed(self, model: str, input: list[str]) -> list[list[float]]:
        """Embeddings de uma lista de textos."""
        r = self._chamar(lambda: self._cliente.embed(model=model, input=list(input)), model)
        vetores = [list(v) for v in r.embeddings]
        if len(vetores) != len(input):
            raise RuntimeError(f"/api/embed devolveu {len(vetores)} vetores para {len(input)} textos.")
        return vetores

    def chat(
        self,
        model: str,
        messages: list[dict],
        options: dict,
        think: bool | None = None,
        format: str | None = None,
    ) -> dict:
        """Uma rodada de chat sem streaming."""
        kwargs: dict = {"model": model, "messages": messages, "options": options, "stream": False}
        if format:
            kwargs["format"] = format
        if think is not None and model not in self._sem_think:
            kwargs["think"] = think
        try:
            r = self._chamar(lambda: self._cliente.chat(**kwargs), model)
        except self._ollama.ResponseError as e:
            # Modelos sem suporte a raciocínio (ex.: llama3.2) rejeitam o parâmetro think.
            if "think" in kwargs and "think" in str(e).lower():
                self._sem_think.add(model)
                kwargs.pop("think")
                r = self._chamar(lambda: self._cliente.chat(**kwargs), model)
            else:
                raise
        return {
            "content": r.message.content or "",
            "model": r.model or model,
            "prompt_eval_count": r.prompt_eval_count,
            "eval_count": r.eval_count,
        }

    def modelos_disponiveis(self) -> list[str]:
        """Nomes dos modelos instalados (``ollama list``)."""
        r = self._chamar(lambda: self._cliente.list())
        return [m.model for m in r.models if m.model]

    def versao(self) -> str:
        """Versão do servidor (``/api/version``) ou ``'?'``."""
        import httpx

        try:
            return httpx.get(f"{self.host}/api/version", timeout=10).json().get("version", "?")
        except (httpx.HTTPError, ValueError):
            return "?"


def criar_cliente(cfg: Config) -> OllamaReal:
    """Cliente real a partir de ``cfg.ollama``."""
    return OllamaReal(cfg.ollama.host, cfg.ollama.timeout_s)


# ---------------------------------------------------------------------------
# Implementação falsa (testes)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[^\W_]{2,}")


def vetor_hash(texto: str, dimensao: int) -> list[float]:
    """Embedding determinístico: contagem de tokens espalhada por hash (com semântica léxica)."""
    v = [0.0] * dimensao
    for tok in _TOKEN.findall(unicodedata.normalize("NFC", texto.lower())):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dimensao] += 1.0
    if not any(v):  # texto sem tokens: um componente fixo derivado do hash do texto
        v[int(hashlib.md5(texto.encode()).hexdigest(), 16) % dimensao] = 1.0
    return v


class OllamaFake:
    """Cliente falso: embeddings por hash e chat com texto fixo ou calculado por uma função.

    Args:
        dimensao: dimensão dos vetores devolvidos por ``embed``.
        resposta: texto fixo ou função ``(modelo, prompt) -> texto`` para o chat.
        modelos: nomes devolvidos por ``modelos_disponiveis``.
    """

    def __init__(
        self,
        dimensao: int = 1024,
        resposta: str | Callable[[str, str], str] = "Não encontrado no manual.",
        modelos: list[str] | None = None,
    ):
        self.host = "fake://ollama"
        self.dimensao = dimensao
        self.resposta = resposta
        self.modelos = (
            modelos
            if modelos is not None
            else [
                "qwen3:4b",
                "qwen3-manual:4b",
                "bge-m3:latest",
                "llama3.2:3b",
            ]
        )
        self.chamadas_chat: list[dict] = []
        self.n_textos_embed = 0

    def embed(self, model: str, input: list[str]) -> list[list[float]]:
        """Vetores de :func:`vetor_hash`."""
        self.n_textos_embed += len(input)
        return [vetor_hash(t, self.dimensao) for t in input]

    def chat(
        self,
        model: str,
        messages: list[dict],
        options: dict,
        think: bool | None = None,
        format: str | None = None,
    ) -> dict:
        """Registra a chamada e devolve o texto fixo ou o da função ``resposta``."""
        prompt = messages[-1]["content"]
        self.chamadas_chat.append(
            {"model": model, "prompt": prompt, "options": dict(options), "think": think, "format": format}
        )
        texto = self.resposta(model, prompt) if callable(self.resposta) else self.resposta
        return {
            "content": texto,
            "model": model,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": len(texto.split()),
        }

    def modelos_disponiveis(self) -> list[str]:
        """Modelos configurados no construtor."""
        return list(self.modelos)

    def versao(self) -> str:
        """Versão fictícia."""
        return "fake"
