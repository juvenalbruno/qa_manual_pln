#!/usr/bin/env python3
"""Avaliação do sistema de QA (arquivo separado do fluxo principal).

Lê o ``resposta.json`` revisado, usa como gabarito só as respostas com ``"aprovada": true`` e roda cada pergunta nos
quatro modos de busca (S0 sem busca, S1 BM25, S2 denso, S3 híbrido), com um ou mais modelos leitores. Para cada
combinação calcula:

    em            Exact Match: a resposta normalizada é igual ao gabarito (0 ou 1).
    f1            F1 de palavras entre resposta e gabarito (0 a 1), estilo SQuAD.
    recall_at_5   o trecho de onde saiu o gabarito está entre os 5 trechos buscados (só perguntas com resposta).
    mrr_at_5      1 / posição desse trecho na busca (0 se ausente).
    abst_correta  fração das perguntas sem resposta em que o sistema disse "Não encontrado no manual".
    abst_indevida fração das perguntas com resposta em que o sistema disse "Não encontrado no manual".
    citacao_valida fração das respostas (não abstidas) cuja citação aponta para um trecho buscado.
    latencia_mediana_s tempo mediano por pergunta.

Uso:
    python avaliar.py                                  # resposta.json, S0 a S3, leitor qwen3:4b
    python avaliar.py --modos S0 S3 --limite 30        # experimento mínimo
    python avaliar.py --modelos qwen3:4b outro:modelo  # compara leitores

Saída em ``resultados/<data-hora>/``: ``respostas.jsonl`` (cada resposta), ``metrics.csv`` (uma linha por modo e
modelo) e ``grafico.png`` (se o matplotlib estiver instalado).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import string
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Usa as funções do fluxo principal (busca, resposta, verificação), que está na mesma pasta.
import qa_manual as qa

# Artigos e preposições ignorados na comparação de respostas (não mudam o sentido).
PALAVRAS_IGNORADAS = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "ao", "à", "aos", "às", "e",
}  # fmt: skip


# ======================================================================================================================
# GABARITO
# ======================================================================================================================


def carregar_gabarito(caminho: Path, indice: qa.Indice) -> list[dict]:
    """Lê o resposta.json e devolve só os itens aprovados, prontos para a avaliação.

    Entradas:
        caminho: arquivo resposta.json revisado.
        indice: índice do manual (para conferir o trecho de origem).
    Saídas:
        lista de {"id", "pergunta", "referencia", "sem_resposta", "evidencia"}, em que ``evidencia`` é a lista de
        ids de trecho onde está a resposta (vazia para perguntas sem resposta).
    Erros:
        ErroUsuario se o arquivo não existir ou nenhum item estiver aprovado.
    """
    if not caminho.exists():
        raise qa.ErroUsuario(f"{caminho.name} não encontrado; rode antes: python qa_manual.py responder")
    itens = json.loads(caminho.read_text(encoding="utf-8"))
    gabarito = []
    for item in itens:
        # Só entram as respostas que o grupo aprovou.
        if item.get("aprovada") is not True:
            continue
        sem_resposta = item.get("tipo") == "sem_resposta" or qa.absteve(item.get("resposta", ""))
        evidencia = []
        if not sem_resposta:
            # O trecho de origem, se ainda existir no índice; senão, os trechos da seção e página indicadas.
            if item.get("trecho") in indice.por_id:
                evidencia = [item["trecho"]]
            else:
                evidencia = [
                    t["id"]
                    for t in indice.trechos
                    if t["secao_id"] == str(item.get("secao"))
                    and (item.get("pagina") is None or t["pagina_inicio"] <= item["pagina"] <= t["pagina_fim"])
                ]
        gabarito.append(
            {
                "id": str(item["id"]),
                "pergunta": item["pergunta"],
                "referencia": qa.FRASE_ABSTENCAO if sem_resposta else item["resposta"],
                "sem_resposta": sem_resposta,
                "evidencia": evidencia,
            }
        )
    if not gabarito:
        raise qa.ErroUsuario(f'nenhuma resposta aprovada em {caminho.name}; marque "aprovada": true nas corretas')
    return gabarito


# ======================================================================================================================
# MÉTRICAS
# ======================================================================================================================


def normalizar(texto: str) -> list[str]:
    """Prepara uma resposta para comparação.

    Entradas:
        texto: resposta do sistema ou do gabarito.
    Saídas:
        lista de palavras em minúsculas, sem a citação "(seção X, p. N)", sem pontuação e sem artigos/preposições.
    """
    # Remove a citação de fonte, que não faz parte do conteúdo da resposta.
    texto = re.sub(r"\(?\s*se[çc][ãa]o[^()]*\)?", " ", texto, flags=re.I)
    # Minúsculas e pontuação trocada por espaço.
    texto = texto.lower().translate(str.maketrans({c: " " for c in string.punctuation + "–—“”"}))
    return [p for p in texto.split() if p not in PALAVRAS_IGNORADAS]


def exact_match(resposta: str, referencia: str) -> int:
    """1 se resposta e gabarito ficam iguais depois de normalizados, senão 0."""
    return int(normalizar(resposta) == normalizar(referencia))


def f1(resposta: str, referencia: str) -> float:
    """F1 de palavras (estilo SQuAD) entre a resposta e o gabarito.

    Entradas:
        resposta: texto do sistema.
        referencia: texto do gabarito.
    Saídas:
        valor de 0 a 1; 1 se as duas ficarem vazias depois de normalizadas.
    """
    r, g = normalizar(resposta), normalizar(referencia)
    if not r or not g:
        return float(r == g)
    # Palavras em comum, contando repetições.
    comuns = sum((Counter(r) & Counter(g)).values())
    if comuns == 0:
        return 0.0
    precisao, cobertura = comuns / len(r), comuns / len(g)
    return 2 * precisao * cobertura / (precisao + cobertura)


def media(valores: list[float]) -> float | None:
    """Média de uma lista (None se vazia)."""
    return round(statistics.mean(valores), 3) if valores else None


def resumir(registros: list[dict]) -> dict:
    """Calcula as métricas de um conjunto de respostas (um modo + um modelo).

    Entradas:
        registros: respostas com os campos gravados em ``avaliar`` (inclui o gabarito de cada pergunta).
    Saídas:
        dicionário com n, em, f1, recall_at_5, mrr_at_5, abst_correta, abst_indevida, citacao_valida e
        latencia_mediana_s.
    """
    com_resposta = [r for r in registros if not r["sem_resposta"]]
    sem_resposta = [r for r in registros if r["sem_resposta"]]
    # Recall e MRR só fazem sentido quando houve busca e quando se sabe onde está a resposta.
    com_busca = [r for r in com_resposta if r["modo"] != "nenhum" and r["evidencia"]]
    nao_abstidas = [r for r in registros if not r["absteve"]]
    return {
        "n": len(registros),
        # Quando o sistema se absteve numa pergunta com resposta, EM e F1 valem 0.
        "em": media([0 if r["absteve"] else exact_match(r["resposta"], r["referencia"]) for r in com_resposta]),
        "f1": media([0.0 if r["absteve"] else f1(r["resposta"], r["referencia"]) for r in com_resposta]),
        "recall_at_5": media([int(bool(set(r["trechos"]) & set(r["evidencia"]))) for r in com_busca]),
        "mrr_at_5": media(
            [next((1 / pos for pos, t in enumerate(r["trechos"], 1) if t in r["evidencia"]), 0.0) for r in com_busca]
        ),
        "abst_correta": media([int(r["absteve"]) for r in sem_resposta]),
        "abst_indevida": media([int(r["absteve"]) for r in com_resposta]),
        "citacao_valida": media([int(r["citacao_valida"]) for r in nao_abstidas]),
        "latencia_mediana_s": round(statistics.median(r["latencia_s"] for r in registros), 2),
    }


# ======================================================================================================================
# EXECUÇÃO
# ======================================================================================================================


def avaliar(gabarito: list[dict], indice: qa.Indice, modos: list[str], modelos: list[str], pasta: Path) -> list[dict]:
    """Roda todas as perguntas em todos os modos e modelos e calcula as métricas.

    Entradas:
        gabarito: saída de ``carregar_gabarito``.
        indice: índice do manual.
        modos: lista entre S0, S1, S2 e S3.
        modelos: modelos leitores a comparar.
        pasta: onde gravar respostas.jsonl.
    Saídas:
        uma linha de métricas por (modo, modelo).
    """
    cliente = qa.criar_cliente()
    linhas = []
    total = len(modos) * len(modelos) * len(gabarito)
    feitos = 0
    with open(pasta / "respostas.jsonl", "w", encoding="utf-8") as arquivo:
        for modelo in modelos:
            for nome_modo in modos:
                registros = []
                for item in gabarito:
                    feitos += 1
                    print(f"  [{feitos}/{total}] {nome_modo} {modelo} pergunta {item['id']}", file=sys.stderr)
                    # Responde com o modo e o modelo da vez e guarda junto o gabarito da pergunta.
                    resultado = qa.responder(item["pergunta"], indice, cliente, qa.MODOS[nome_modo], modelo)
                    registro = {
                        "modo_exp": nome_modo,
                        "modo": qa.MODOS[nome_modo],
                        "modelo": modelo,
                        **item,
                        **resultado,
                    }
                    registros.append(registro)
                    # Grava cada resposta assim que sai (útil para conferir depois).
                    arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    arquivo.flush()
                linhas.append({"modo": nome_modo, "modelo": modelo, **resumir(registros)})
    return linhas


def gravar_metricas(linhas: list[dict], pasta: Path) -> None:
    """Grava metrics.csv e, se o matplotlib existir, um gráfico de barras com EM, F1 e Recall@5 por modo.

    Entradas:
        linhas: saída de ``avaliar``.
        pasta: pasta da execução.
    """
    with open(pasta / "metrics.csv", "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Um grupo de barras por linha (modo + modelo), com EM, F1 e Recall@5.
    rotulos = [f"{linha['modo']}\n{linha['modelo']}" for linha in linhas]
    metricas = ["em", "f1", "recall_at_5"]
    largura = 0.25
    figura, eixo = plt.subplots(figsize=(max(6, 1.4 * len(linhas)), 4))
    for n, metrica in enumerate(metricas):
        valores = [linha[metrica] or 0 for linha in linhas]
        eixo.bar([i + (n - 1) * largura for i in range(len(linhas))], valores, width=largura, label=metrica)
    eixo.set_xticks(range(len(linhas)), rotulos)
    eixo.set_ylim(0, 1)
    eixo.legend()
    eixo.set_title("Métricas por modo de busca")
    figura.tight_layout()
    figura.savefig(pasta / "grafico.png", dpi=150)
    plt.close(figura)


def imprimir_tabela(linhas: list[dict]) -> None:
    """Mostra as métricas no terminal em formato de tabela simples."""
    colunas = list(linhas[0])
    larguras = [max(len(c), *(len(str(linha[c])) for linha in linhas)) for c in colunas]
    print("  ".join(c.ljust(w) for c, w in zip(colunas, larguras, strict=True)))
    for linha in linhas:
        print("  ".join(str(linha[c]).ljust(w) for c, w in zip(colunas, larguras, strict=True)))


def main(argv: list[str] | None = None) -> int:
    """Lê os argumentos, roda a avaliação e grava os resultados.

    Saídas:
        código de saída: 0 sucesso, 2 erro de uso, 130 interrompido.
    """
    parser = argparse.ArgumentParser(prog="avaliar.py", description="Métricas do sistema de QA.")
    parser.add_argument("--respostas", default=str(qa.ARQ_RESPOSTAS), help="resposta.json revisado")
    parser.add_argument("--modos", nargs="+", choices=qa.MODOS, default=list(qa.MODOS), help="padrão: S0 S1 S2 S3")
    parser.add_argument("--modelos", nargs="+", default=[qa.MODELO_LEITOR], help="leitores a comparar")
    parser.add_argument("--limite", type=int, help="usa só as N primeiras perguntas aprovadas")
    args = parser.parse_args(argv)
    try:
        indice = qa.carregar_indice()
        gabarito = carregar_gabarito(Path(args.respostas), indice)[: args.limite]
        # Cada execução ganha uma pasta própria com data e hora.
        pasta = qa.RAIZ / "resultados" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pasta.mkdir(parents=True)
        print(f"Avaliando {len(gabarito)} perguntas aprovadas ...", file=sys.stderr)
        linhas = avaliar(gabarito, indice, args.modos, args.modelos, pasta)
        gravar_metricas(linhas, pasta)
        print()
        imprimir_tabela(linhas)
        print(f"\nResultados em {pasta}/")
        return 0
    except qa.ErroUsuario as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrompido.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
