import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from tests.fake_ollama import FakeOllama  # noqa: E402


@pytest.fixture(scope="session")
def fake_ollama():
    with FakeOllama() as f:
        yield f


@pytest.fixture
def projeto(tmp_path, monkeypatch, fake_ollama):
    """Diretório de trabalho isolado com o manual de exemplo e o Ollama falso."""
    from gerar_manual_exemplo import gerar

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_HOST", fake_ollama.url)
    gerar(tmp_path / "data" / "manual.pdf")
    return tmp_path
