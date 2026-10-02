"""Passo 14: perplexidade dos leitores base e ajustado sobre trechos reservados, com ``llama-perplexity``."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from .io_utils import ErroUsuario

log = logging.getLogger(__name__)

_PPL_FINAL = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)")
_FROM = re.compile(r"^FROM\s+(\S.*)$", re.MULTILINE)


def rodar_llama_perplexity(binario: Path, gguf: Path, texto: Path, ctx: int) -> float:
    """Executa ``{binario} -m {gguf} -f {texto} -c {ctx}`` e devolve o valor de ``Final estimate: PPL =``.

    Todos os blocos de ``ctx`` tokens do texto são avaliados (padrão ``--chunks -1`` do llama.cpp; ``--chunks 0``
    avaliaria zero blocos). O texto precisa ter pelo menos ``2 * ctx`` tokens.

    Raises:
        ErroUsuario: se o binário, o GGUF ou o texto não existirem, ou se a execução falhar.
    """
    for rotulo, caminho in (("binário llama-perplexity", binario), ("GGUF", gguf), ("texto", texto)):
        if not Path(caminho).is_file():
            raise ErroUsuario(f"{rotulo} não encontrado: {caminho}")
    cmd = [str(binario), "-m", str(gguf), "-f", str(texto), "-c", str(ctx)]
    log.info("llama-perplexity: modelo %s, ctx %d", Path(gguf).name, ctx)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        raise ErroUsuario(f"não foi possível executar {binario}: {e.strerror}") from None
    saida = (r.stdout or "") + "\n" + (r.stderr or "")
    m = _PPL_FINAL.search(saida)
    if r.returncode != 0 or not m:
        dicas = [ln.strip() for ln in saida.splitlines() if re.search(r"error|need at least|failed", ln, re.I)]
        detalhe = "; ".join(dicas[-3:]) or f"código de saída {r.returncode}"
        raise ErroUsuario(
            f"llama-perplexity falhou para {Path(gguf).name}: {detalhe}. "
            f"O texto precisa de pelo menos {2 * ctx} tokens; reserve mais trechos ou reduza avaliacao.ppl_ctx."
        )
    return float(m.group(1))


def gguf_do_ollama(modelo: str) -> Path:
    """Caminho do GGUF de um modelo do Ollama, lido da linha ``FROM`` de ``ollama show <modelo> --modelfile``.

    O arquivo blob apontado (``.../blobs/sha256-...``) é um GGUF válido para o llama.cpp.
    """
    if shutil.which("ollama") is None:
        raise ErroUsuario("comando `ollama` não encontrado no PATH; informe o GGUF com --gguf-base <arquivo>.")
    r = subprocess.run(["ollama", "show", modelo, "--modelfile"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise ErroUsuario(
            f"`ollama show {modelo} --modelfile` falhou; o modelo está instalado? (`ollama pull {modelo}`)"
        )
    for caminho in _FROM.findall(r.stdout):
        p = Path(caminho.strip())
        if p.is_file():
            return p
    raise ErroUsuario(f"não encontrei o arquivo GGUF de {modelo} na linha FROM do Modelfile.")
