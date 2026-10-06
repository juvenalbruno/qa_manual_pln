from qa_manual.tokenize_pt import remover_acentos, tokenizar


def test_minusculas_pontuacao_e_stopwords():
    assert tokenizar("Qual é o PRAZO para a liberação da carga?") == ["prazo", "liberação", "carga"]


def test_acentos_mantidos_ou_removidos():
    assert "liberação" in tokenizar("Liberação")
    assert tokenizar("Liberação da carga", manter_acentos=False) == ["liberacao", "carga"]
    assert remover_acentos("ção") == "cao"


def test_tokens_de_um_caractere_e_sem_stopwords():
    assert tokenizar("a b 5 km/h", remover_stopwords=False) == ["km"]
    assert tokenizar("de da do", remover_stopwords=False) == ["de", "da", "do"]
    assert tokenizar("de da do") == []


def test_numeros_e_simbolos():
    assert tokenizar("Berço 202: 11,5 m (NR-29)") == ["berço", "202", "11", "nr", "29"]
