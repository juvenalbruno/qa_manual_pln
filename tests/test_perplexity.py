from __future__ import annotations

import stat

import pytest

from qa_manual.io_utils import ErroUsuario
from qa_manual.perplexity import rodar_llama_perplexity


def _binario(tmp_path, corpo: str):
    b = tmp_path / "llama-perplexity"
    b.write_text("#!/bin/sh\n" + corpo + "\n")
    b.chmod(b.stat().st_mode | stat.S_IEXEC)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    texto = tmp_path / "ppl.txt"
    texto.write_text("texto fictício\n")
    return b, gguf, texto


def test_le_o_valor_final(tmp_path):
    b, gguf, texto = _binario(tmp_path, 'echo "[1]6.1,[2]7.0" >&2; echo "Final estimate: PPL = 7.1234 +/- 0.05123" >&2')
    assert rodar_llama_perplexity(b, gguf, texto, 2048) == pytest.approx(7.1234)


def test_texto_curto_e_arquivos_ausentes(tmp_path):
    b, gguf, texto = _binario(tmp_path, 'echo "error: you need at least 4096 tokens to evaluate" >&2; exit 1')
    with pytest.raises(ErroUsuario, match="need at least"):
        rodar_llama_perplexity(b, gguf, texto, 2048)
    with pytest.raises(ErroUsuario, match="GGUF não encontrado"):
        rodar_llama_perplexity(b, tmp_path / "x.gguf", texto, 2048)
    with pytest.raises(ErroUsuario, match="binário"):
        rodar_llama_perplexity(tmp_path / "nada", gguf, texto, 2048)
