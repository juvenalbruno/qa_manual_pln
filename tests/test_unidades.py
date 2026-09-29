import pytest

from src.analysis import bootstrap_diferenca, mapear_evidencias, mcnemar_exato, sugerir_categoria
from src.evaluate import (
    agregar_repeticoes,
    exact_match,
    f1_tokens,
    kappa_cohen,
    kappa_ponderado,
    metricas,
    percentil,
    pontuar,
    recall_em_k,
    rr,
)
from src.generate import extrair_citacao
from src.gold import amostrar_estratificado, dividir_por_secao, subconjunto_comum
from src.ingest import Elemento, Linha, _continua_titulo, _juntar_linhas, _normalizar_unicode, eh_titulo, segmentar
from src.ollama_client import validar_url_local
from src.retrieve import rrf
from src.utils import ErroUsuario, contem_abstencao, normalizar_resposta, renderizar, tokenizar_bm25


def test_normalizacao_remove_artigos_preposicoes_pontuacao():
    assert normalizar_resposta("Até 48 horas, após a atracação!") == "até 48 horas após atracação"
    assert normalizar_resposta("  O   prazo  do navio ") == "prazo navio"


def test_em_e_f1():
    assert exact_match("Até 48 horas após a atracação.", "até 48 horas após atracação") == 1.0
    # referência normalizada: "até 48 horas após atracação" (5 tokens) → P = 1, R = 2/5
    assert f1_tokens("48 horas", "até 48 horas após a atracação") == pytest.approx(2 * 0.4 / 1.4)
    assert f1_tokens("", "algo") == 0.0


def test_tokenizacao_bm25_sem_acentos_e_stopwords():
    assert tokenizar_bm25("Qual é o prazo de liberação da carga?") == ["prazo", "liberacao", "carga"]


def test_rrf():
    fundido = rrf([[("a", 9.0), ("b", 5.0)], [("b", 0.9), ("c", 0.8)]], 60)
    assert fundido[0][0] == "b"
    assert fundido[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_recall_e_mrr():
    assert recall_em_k(["x", "p1", "p2"], ["p1"], 1) == 0.0
    assert recall_em_k(["x", "p1", "p2"], ["p1"], 2) == 1.0
    assert rr(["x", "p1"], ["p1"]) == 0.5
    assert recall_em_k(["x"], [], 5) is None
    assert recall_em_k(["p2", "x"], ["p1", "p2"], 1) == 1.0  # evidências alternativas


def test_extrair_citacao():
    curta, secao, pagina = extrair_citacao("A carga é liberada em até 48 horas (seção 4.1 Prazos, p. 17).")
    assert curta.rstrip(".") == "A carga é liberada em até 48 horas"
    assert secao == "4.1 Prazos" and pagina == 17
    assert extrair_citacao("Sem citação.")[1:] == (None, None)


def test_abstencao():
    assert contem_abstencao("Nao encontrado no manual.")
    assert not contem_abstencao("A resposta é 48 horas.")
    assert contem_abstencao("Não sei.", ["não sei"])


def test_renderizar_preserva_json_e_nao_reprocessa():
    tpl = 'Responda em JSON: {"pergunta": "..."}\nTrecho: {texto} / {pergunta}'
    out = renderizar(tpl, texto="valor {pergunta}", pergunta="Q")
    assert out == 'Responda em JSON: {"pergunta": "..."}\nTrecho: valor {pergunta} / Q'


def test_url_ollama_restrita_a_localhost():
    assert validar_url_local("http://localhost:11434/") == "http://localhost:11434"
    with pytest.raises(ErroUsuario):
        validar_url_local("https://api.exemplo.com")


def _linha(texto, tamanho=10.0, negrito=False):
    return Linha(texto=texto, pagina=1, bloco=0, y0=0, y1=0, tamanho=tamanho, negrito=negrito)


@pytest.mark.parametrize(
    "texto,tamanho,negrito,esperado",
    [
        ("4.2 Liberação de carga", 10, False, True),
        ("4 Operação", 14, True, True),
        ("SEGURANÇA DO TRABALHO", 10, False, True),
        ("1. Abra a tampa do compartimento.", 10, False, False),
        ("2.5 kg de carga máxima", 10, False, False),
        ("O operador deve verificar os freios antes do turno.", 10, False, False),
        ("• Verifique o nível de óleo", 10, False, False),
        ("ATENÇÃO", 10, True, False),
        ("Observação:", 10, True, False),
        ("Procedimento de partida:", 10, True, False),
    ],
)
def test_deteccao_de_titulos(texto, tamanho, negrito, esperado):
    assert eh_titulo(_linha(texto, tamanho, negrito), 10.0) is esperado


def test_segmentacao_respeita_limite_secao_e_sobreposicao():
    frase = "Esta é uma frase de teste com exatamente onze palavras no total. "
    elementos = [
        Elemento("titulo", "1 Primeira", 1),
        Elemento("paragrafo", frase * 30, 1),
        Elemento("titulo", "2 Segunda", 2),
        Elemento("paragrafo", frase * 5, 2),
    ]
    ps = segmentar(elementos, tamanho_tokens=100, sobreposicao_tokens=20, palavras_por_token=0.74)
    assert all(p["n_palavras"] <= 74 for p in ps)
    assert {p["secao"] for p in ps} == {"1 Primeira", "2 Segunda"}
    primeira = [p for p in ps if p["secao"] == "1 Primeira"]
    assert len(primeira) > 1
    # a sobreposição repete o fim de uma passagem no início da seguinte
    assert primeira[0]["texto"].split(". ")[-1].strip(". ") in primeira[1]["texto"]
    assert all(p["secao"] == "2 Segunda" for p in ps if p["pagina"] == 2)


def test_kappas():
    assert kappa_cohen([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert kappa_ponderado([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    assert kappa_ponderado([0, 0, 2, 2], [2, 2, 0, 0]) < 0


def test_mcnemar_e_bootstrap():
    assert mcnemar_exato(0, 0) == 1.0
    assert mcnemar_exato(10, 0) < 0.01
    b = bootstrap_diferenca([0.0] * 20, [1.0] * 20)
    assert b["diferenca"] == 1.0 and b["ic_inf"] == 1.0


def test_divisao_por_secao_sem_vazamento():
    secao_de = {f"p{i}": f"S{i // 3}" for i in range(30)}
    itens = [{"evidencia": [f"p{i}"], "tipo": "factual"} for i in range(30)]
    itens.append({"evidencia": ["p1", "p10"], "tipo": "factual"})  # liga S0 e S3
    itens += [{"evidencia": [], "tipo": "sem_resposta"} for _ in range(10)]
    dev, teste = dividir_por_secao(itens, secao_de, 0.3, seed=1)
    secoes = lambda l: {secao_de[e] for it in l for e in it["evidencia"]}  # noqa: E731
    assert not secoes(dev) & secoes(teste)
    assert ("S0" in secoes(dev)) == ("S3" in secoes(dev))
    assert len(dev) + len(teste) == len(itens)
    assert 0.15 < len(dev) / len(itens) < 0.45


def test_amostragem_estratificada_cobre_secoes():
    ps = [{"id": f"p{i}", "secao": f"S{i % 5}", "n_palavras": 100} for i in range(50)]
    amostra = amostrar_estratificado(ps, 10)
    assert len({p["secao"] for p in amostra}) == 5
    assert len({p["id"] for p in amostra}) == 10


def test_subconjunto_comum_deterministico():
    ids = [f"c{i:04d}" for i in range(100)]
    assert subconjunto_comum(ids) == subconjunto_comum(list(reversed(ids)))
    assert len(subconjunto_comum(ids)) == 30


def test_pontuar_e_metricas():
    gold_resp = {"tipo": "factual", "resposta_ref": "48 horas", "evidencia": ["p1"]}
    gold_sem = {"tipo": "sem_resposta", "resposta_ref": "Não encontrado no manual", "evidencia": []}
    reg_ok = {"modo": "bm25", "k": 2, "recuperadas": [{"id": "p1"}, {"id": "p2"}], "absteve": False,
              "resposta": "48 horas (seção X, p. 3).", "resposta_curta": "48 horas", "pagina_citada": 3,
              "latencia_s": 2.0}
    reg_abs = {"modo": "bm25", "k": 2, "recuperadas": [{"id": "p3"}], "absteve": True,
               "resposta": "Não encontrado no manual.", "latencia_s": 4.0}
    por_id = {"p1": {"pagina": 3, "pagina_fim": 3}}
    r1 = {**reg_ok, **pontuar(gold_resp, reg_ok, por_id)}
    r2 = {**reg_abs, **pontuar(gold_sem, reg_abs, por_id)}
    assert r1["em"] == 1.0 and r1["citacao_correta"] == 1.0 and r1["recall"] == 1.0
    assert r2["em"] == 1.0 and r2["recall"] is None
    m = metricas([r1, r2])
    assert m["recall@2"] == 1.0 and m["abst_correta"] == 1.0 and m["abst_indevida"] == 0.0
    assert m["latencia_mediana_s"] == 3.0
    agg = agregar_repeticoes([m, {**m, "f1": 0.0}])
    assert agg["f1"] == 0.5 and agg["desvio"]["f1"] > 0 and agg["repeticoes"] == 2


def test_percentil():
    assert percentil([1, 2, 3, 4, 5], 0.5) == 3
    assert percentil([], 0.5) is None


def test_categoria_sugerida():
    base = {"tipo": "factual", "modo": "hibrido", "absteve": False, "fidelidade": 1}
    assert sugerir_categoria({**base, "recall": 0.0}) == "falha_recuperacao"
    assert sugerir_categoria({**base, "recall": 1.0, "absteve": True}) == "abstencao_indevida"
    assert sugerir_categoria({**base, "recall": 1.0, "fidelidade": 0}) == "alucinacao"
    assert sugerir_categoria({**base, "recall": 1.0}) == "falha_leitura"
    assert sugerir_categoria({**base, "tipo": "sem_resposta"}) == "alucinacao"


def test_mapear_evidencias_entre_segmentacoes():
    base = {"b1": {"secao": "S", "texto": "alfa beta gama delta epsilon zeta eta teta"}}
    nova = [
        {"id": "n1", "secao": "S", "texto": "alfa beta gama delta"},
        {"id": "n2", "secao": "S", "texto": "iota kappa lambda mu"},
        {"id": "n3", "secao": "T", "texto": "alfa beta gama delta"},
    ]
    assert mapear_evidencias([{"id": "q1", "evidencia": ["b1"]}], base, nova) == {"q1": ["n1"]}


def test_normalizacao_unicode_preserva_simbolos_e_desfaz_ligaduras():
    assert _normalizar_unicode("Portaria nº 671, 76,2 m³, reﬂetivo") == "Portaria nº 671, 76,2 m³, refletivo"


def test_juntar_linhas():
    assert _juntar_linhas(["velocidade de 15 km/", "h no pátio"]) == "velocidade de 15 km/h no pátio"
    assert _juntar_linhas(["manu-", "tenção preventiva"]) == "manutenção preventiva"
    assert _juntar_linhas(["Itens:", "• capacete"]) == "Itens:\n• capacete"


def test_titulo_em_duas_linhas():
    a = Linha("Manual de Operações do Porto", 1, 0, 189, 219, 22.0, True)
    b = Linha("de Salvador", 1, 1, 216, 247, 22.0, True)
    c = Linha("2.1 Localização", 1, 2, 250, 265, 22.0, True)
    assert _continua_titulo(a, b)
    assert not _continua_titulo(b, c)  # nova numeração inicia outro título
