from __future__ import annotations

import pytest

from qa_manual.chunking import Trecho
from qa_manual.verify import detectar_abstencao, extrair_citacoes, remover_citacoes, validar_fonte

FRASE = "Não encontrado no manual"


@pytest.mark.parametrize(
    "resposta",
    [
        "Não encontrado no manual.",
        "NÃO ENCONTRADO NO MANUAL",
        "nao encontrado no manual!",
        "Resposta: Não encontrado no manual.",
    ],
)
def test_abstencao_detectada(resposta):
    assert detectar_abstencao(resposta, FRASE)


def test_abstencao_nao_detectada():
    assert not detectar_abstencao("A carga é liberada em 48 horas (seção 2.2, p. 5).", FRASE)
    assert not detectar_abstencao("Não encontrado.", FRASE)


def test_extrair_citacoes_em_quatro_formatos():
    assert extrair_citacoes("A carga sai em 48 horas (seção 4.2, p. 17).") == [{"tema_id": "4.2", "pagina": 17}]
    assert extrair_citacoes("Ver seção 4.2.") == [{"tema_id": "4.2", "pagina": None}]
    assert extrair_citacoes("Conforme p. 17 do manual.") == [{"tema_id": None, "pagina": 17}]
    assert extrair_citacoes("Está na página 17.") == [{"tema_id": None, "pagina": 17}]
    assert extrair_citacoes("Na pág. 9 e na (seção {3.1}, p. {12}).") == [
        {"tema_id": None, "pagina": 9},
        {"tema_id": "3.1", "pagina": 12},
    ]
    assert extrair_citacoes("Sem citação alguma; capítulo cap. 3.") == []


def _trecho(tema_id, inicio, fim):
    return Trecho(
        id="t0001",
        tema=f"{tema_id} X",
        tema_id=tema_id,
        pagina_inicio=inicio,
        pagina_fim=fim,
        paragrafos=1,
        n_tokens=10,
        texto="...",
    )


def test_validar_fonte():
    recuperados = [_trecho("4.2", 17, 18), _trecho("2.1", 3, 3)]
    assert validar_fonte([{"tema_id": "4.2", "pagina": 18}], recuperados)
    assert validar_fonte([{"tema_id": None, "pagina": 3}], recuperados)
    assert not validar_fonte([{"tema_id": "4.2", "pagina": 3}], recuperados)
    assert not validar_fonte([{"tema_id": "9.9", "pagina": None}], recuperados)
    assert not validar_fonte([{"tema_id": None, "pagina": None}], recuperados)
    assert not validar_fonte([{"tema_id": "4.2", "pagina": 17}], [])  # S0


def test_remover_citacoes():
    assert remover_citacoes("A carga sai em 48 horas (seção 4.2, p. 17).") == "A carga sai em 48 horas."
    assert remover_citacoes("72 horas, seção 2.1, p. 3") == "72 horas,"
    assert remover_citacoes("Sem citação.") == "Sem citação."
