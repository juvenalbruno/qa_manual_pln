"""O notebook do Colab é autossuficiente e está em dia com o código do projeto."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from qa_manual.io_utils import RAIZ

_spec = importlib.util.spec_from_file_location("gerar_notebook", RAIZ / "colab" / "gerar_notebook.py")
gerador = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gerador)


def test_notebook_em_dia_com_o_codigo():
    atual = gerador.NOTEBOOK.read_text(encoding="utf-8")
    assert atual == gerador.serializar(gerador.construir()), "rode: python colab/gerar_notebook.py"


def test_nada_vem_do_repositorio():
    nb = json.loads(gerador.NOTEBOOK.read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for proibido in ("git clone", "github.com", "REPO_URL", "pip install -q --no-deps -e"):
        assert proibido not in codigo, proibido
    # O código próprio do notebook (fora dos arquivos embutidos) nunca abre o manual nem os trechos dele.
    proprio = "\n".join(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and not "".join(c["source"]).startswith("%%writefile")
    )
    assert "data/manual.pdf" not in proprio and "data/trechos.jsonl" not in proprio


def test_codigo_embutido_roda_sozinho(tmp_path):
    """Simula o Colab: grava só os arquivos embutidos e importa o pacote a partir deles."""
    for caminho, conteudo in gerador.arquivos_embutidos().items():
        destino = tmp_path / caminho
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(conteudo, encoding="utf-8")
    pdf = RAIZ / "docs" / "exemplos" / "porto_salvador" / "manual_porto_salvador.pdf"
    script = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
import qa_manual
assert qa_manual.__file__.startswith({str(tmp_path)!r}), qa_manual.__file__
from qa_manual import chunking, config, gold, ingest, prompts
from qa_manual.generate import montar_prompt
from qa_manual.verify import remover_citacoes
cfg = config.carregar()
trechos = chunking.agrupar_trechos(ingest.construir_paragrafos({str(pdf)!r}, cfg), cfg)
prompt = montar_prompt("Qual a profundidade do Berço 202?", trechos[:3])
assert "Trechos:" in prompt and "(seção {{tema_id}}, p. {{pagina}})" in prompt
prompts.preencher(prompts.carregar("gerador_pares_treino"), tipo="factual", tema_id="1", pagina=1, texto="x")
assert gold.extrair_json('ok {{"pergunta": "P?"}}') == {{"pergunta": "P?"}}
print(len(trechos), remover_citacoes("11,5 m (seção 2.3, p. 6)"))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    n, resto = r.stdout.split(" ", 1)
    assert int(n) > 10 and resto.strip() == "11,5 m"
