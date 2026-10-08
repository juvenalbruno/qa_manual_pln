#!/usr/bin/env python3
"""Avaliação do sistema de QA (arquivo separado do fluxo principal).

Lê o ``resposta.json`` revisado, usa como gabarito só as respostas com ``"aprovada": true`` e roda cada pergunta nos
quatro modos de busca (S0 sem busca, S1 BM25, S2 denso, S3 híbrido), com um ou mais modelos leitores. Para cada
combinação calcula:

    em            Exact Match: a resposta normalizada é igual ao gabarito (0 ou 1).
    f1            F1 de palavras entre resposta e gabarito (0 a 1), estilo SQuAD.
    bertscore_f1  semelhança de sentido entre resposta e gabarito (0 a 1), medida pelo BERTimbau; aceita
                  respostas certas escritas com outras palavras, que o EM e o F1 punem.
    recall_at_5   o trecho de onde saiu o gabarito está entre os 5 trechos buscados (só perguntas com resposta).
    mrr_at_5      1 / posição desse trecho na busca (0 se ausente).
    abst_correta  fração das perguntas sem resposta em que o sistema disse "Não encontrado no manual".
    abst_indevida fração das perguntas com resposta em que o sistema disse "Não encontrado no manual".
    citacao_valida fração das respostas (não abstidas) cuja citação aponta para um trecho buscado.
    latencia_mediana_s tempo mediano por pergunta.
    perplexidade  o quanto o modelo leitor "se surpreende" com o texto do manual (quanto menor, mais familiar);
                  calculada pelo llama-perplexity do llama.cpp, uma vez por modelo (igual para todos os modos).

Uso:
    python avaliar.py                                  # resposta.json, S0 a S3, leitor qwen3:4b
    python avaliar.py --modos S0 S3 --limite 30        # experimento mínimo
    python avaliar.py --modelos qwen3:4b outro:modelo  # compara leitores
    python avaliar.py --sem-bertscore                  # pula o BERTScore (mais rápido)
    python avaliar.py --sem-perplexidade               # pula a perplexidade
    python avaliar.py --llama-perplexity /caminho/llama-perplexity   # se o llama.cpp não estiver em ~/llama.cpp

Saída em ``resultados/<data-hora>/``: ``respostas.jsonl`` (cada resposta), ``metrics.csv`` (uma linha por modo e
modelo), ``grafico.png`` e ``grafico_perplexidade.png`` (se o matplotlib estiver instalado) e ``ppl_texto.txt`` (o
texto usado na perplexidade).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
import string
import subprocess
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
        "bertscore_f1": media([r["bertscore"] for r in com_resposta if "bertscore" in r]),
        "recall_at_5": media([int(bool(set(r["trechos"]) & set(r["evidencia"]))) for r in com_busca]),
        "mrr_at_5": media(
            [
                next(
                    (1 / pos for pos, t in enumerate(r["trechos"], 1) if t in r["evidencia"]),
                    0.0,
                )
                for r in com_busca
            ]
        ),
        "abst_correta": media([int(r["absteve"]) for r in sem_resposta]),
        "abst_indevida": media([int(r["absteve"]) for r in com_resposta]),
        "citacao_valida": media([int(r["citacao_valida"]) for r in nao_abstidas]),
        "latencia_mediana_s": round(statistics.median(r["latencia_s"] for r in registros), 2),
    }


# Modelo BERT em português usado pelo BERTScore e a camada de onde saem as representações (recomendação do
# bert-score para modelos BERT-base).
MODELO_BERTSCORE = "neuralmind/bert-base-portuguese-cased"
CAMADAS_BERTSCORE = 9


def criar_bertscore():
    """Carrega o BERTScore com o BERTimbau, uma vez só, para todas as respostas.

    Entradas:
        nenhuma.
    Saídas:
        objeto ``BERTScorer`` pronto para comparar respostas, ou None se o pacote bert-score não estiver instalado.
        Na primeira vez o BERTimbau (~430 MB) é baixado do Hugging Face: é download de modelo, nenhum texto do
        manual é enviado.
    """
    try:
        from bert_score import BERTScorer
    except ImportError:
        print(
            "aviso: bert-score não instalado; a coluna bertscore_f1 ficará vazia (pip install bert-score)",
            file=sys.stderr,
        )
        return None
    # Roda em CPU e sem reescala pela linha de base (valores brutos de 0 a 1).
    avaliador = BERTScorer(
        model_type=MODELO_BERTSCORE,
        num_layers=CAMADAS_BERTSCORE,
        lang="pt",
        device="cpu",
    )
    # O tokenizador do BERTimbau declara um tamanho máximo gigante, que quebra o truncamento do bert-score;
    # limita ao tamanho real do BERT (512 tokens).
    avaliador._tokenizer.model_max_length = 512
    return avaliador


def calcular_bertscore(registros: list[dict], avaliador) -> None:
    """Acrescenta o campo ``bertscore`` a cada resposta de pergunta com resposta.

    Entradas:
        registros: respostas de um modo e modelo (com a referência do gabarito).
        avaliador: saída de ``criar_bertscore`` (None = não calcula).
    Saídas:
        nenhuma; altera os registros. Resposta abstida vale 0; perguntas sem resposta não recebem o campo.
    """
    if avaliador is None:
        return
    # Compara de uma vez só (em lote) as respostas não abstidas com as referências, sem a citação de fonte.
    alvos = [r for r in registros if not r["sem_resposta"] and not r["absteve"]]
    if alvos:
        candidatas = [" ".join(normalizar_para_bertscore(r["resposta"])) or "-" for r in alvos]
        referencias = [" ".join(normalizar_para_bertscore(r["referencia"])) or "-" for r in alvos]
        _, _, f1s = avaliador.score(candidatas, referencias)
        for r, valor in zip(alvos, f1s.tolist(), strict=True):
            r["bertscore"] = round(valor, 4)
    # Abstenção numa pergunta com resposta conta como 0, como no EM e no F1.
    for r in registros:
        if not r["sem_resposta"] and r["absteve"]:
            r["bertscore"] = 0.0


def normalizar_para_bertscore(texto: str) -> list[str]:
    """Tira só a citação de fonte (o BERTScore compara o texto como está, com artigos e pontuação).

    Entradas:
        texto: resposta ou referência.
    Saídas:
        lista de palavras sem a citação "(seção X, p. N)".
    """
    return re.sub(r"\(?\s*se[çc][ãa]o[^()]*\)?", " ", texto, flags=re.I).split()


# ======================================================================================================================
# PERPLEXIDADE
# ======================================================================================================================

# Onde o README manda compilar o llama.cpp.
LLAMA_PERPLEXITY = Path("~/llama.cpp/build/bin/llama-perplexity").expanduser()
# Quantos trechos do manual entram no texto medido e a semente do sorteio (mesmo texto a cada execução).
N_TRECHOS_PPL = 30
SEMENTE_PPL = 42
# Janela de contexto do cálculo. O texto precisa ter pelo menos o dobro disso em tokens.
CONTEXTO_PPL = 2048


def montar_texto_ppl(indice: qa.Indice, destino: Path) -> Path:
    """Sorteia trechos do manual e grava o texto usado na perplexidade.

    Entradas:
        indice: índice do manual.
        destino: arquivo de texto a gravar (fica em resultados/, que não vai para o git).
    Saídas:
        o caminho do arquivo gravado (trechos separados por linha em branco).
    """
    # Sorteio com semente fixa e em ordem de id, para o texto ser sempre o mesmo.
    escolhidos = random.Random(SEMENTE_PPL).sample(indice.trechos, min(N_TRECHOS_PPL, len(indice.trechos)))
    escolhidos.sort(key=lambda t: t["id"])
    destino.write_text("\n\n".join(t["texto"] for t in escolhidos) + "\n", encoding="utf-8")
    return destino


def gguf_do_ollama(modelo: str) -> Path:
    """Acha o arquivo GGUF que o Ollama usa para um modelo.

    Entradas:
        modelo: nome no Ollama (ex.: qwen3:4b).
    Saídas:
        caminho do arquivo (o "blob" do Ollama é um GGUF válido para o llama.cpp).
    Erros:
        ErroUsuario se o comando ollama não existir ou o modelo não estiver instalado.
    """
    if shutil.which("ollama") is None:
        raise qa.ErroUsuario("comando ollama não encontrado; informe o arquivo com --gguf modelo=arquivo.gguf")
    # O Modelfile mostrado pelo Ollama tem uma linha "FROM /caminho/do/blob".
    comando = ["ollama", "show", modelo, "--modelfile"]
    saida = subprocess.run(comando, capture_output=True, text=True, check=False)
    if saida.returncode != 0:
        # Modelo ainda não baixado: baixa e tenta de novo.
        subprocess.run(["ollama", "pull", modelo], check=False)
        saida = subprocess.run(comando, capture_output=True, text=True, check=False)
    if saida.returncode != 0:
        raise qa.ErroUsuario(f"modelo {modelo} não encontrado no Ollama")
    for caminho in re.findall(r"^FROM\s+(\S.*)$", saida.stdout, re.M):
        if Path(caminho.strip()).is_file():
            return Path(caminho.strip())
    raise qa.ErroUsuario(f"não encontrei o arquivo GGUF de {modelo} na saída de `ollama show`")


def medir_perplexidade(binario: Path, gguf: Path, texto: Path) -> float:
    """Roda o llama-perplexity e devolve a perplexidade final.

    Entradas:
        binario: caminho do executável llama-perplexity.
        gguf: arquivo do modelo.
        texto: arquivo de texto a medir.
    Saídas:
        valor da perplexidade (quanto menor, melhor).
    Erros:
        ErroUsuario se a execução falhar (por exemplo, texto curto demais para o contexto).
    """
    comando = [str(binario), "-m", str(gguf), "-f", str(texto), "-c", str(CONTEXTO_PPL)]
    saida = subprocess.run(comando, capture_output=True, text=True, check=False)
    # O llama.cpp termina com a linha "Final estimate: PPL = 7.1234 +/- ...".
    resultado = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", saida.stdout + saida.stderr)
    if saida.returncode != 0 or not resultado:
        raise qa.ErroUsuario(
            f"llama-perplexity falhou para {gguf.name}; o texto precisa de pelo menos {2 * CONTEXTO_PPL} tokens"
        )
    return float(resultado.group(1))


def calcular_perplexidades(
    modelos: list[str],
    indice: qa.Indice,
    binario: Path,
    ggufs: dict[str, str],
    pasta: Path,
) -> dict[str, float | None]:
    """Mede a perplexidade de cada modelo leitor sobre o mesmo texto do manual.

    Entradas:
        modelos: modelos leitores da avaliação.
        indice: índice do manual (fonte do texto).
        binario: caminho do llama-perplexity.
        ggufs: arquivos GGUF informados à mão ({modelo: caminho}); os demais vêm do Ollama.
        pasta: pasta da execução (onde fica o ppl_texto.txt).
    Saídas:
        {modelo: perplexidade}; None quando não deu para medir (com aviso no terminal, sem parar a avaliação).
    """
    if not binario.is_file():
        print(
            f"aviso: llama-perplexity não encontrado em {binario}; a coluna perplexidade ficará vazia "
            "(veja a compilação do llama.cpp no README ou use --llama-perplexity)",
            file=sys.stderr,
        )
        return {}
    texto = montar_texto_ppl(indice, pasta / "ppl_texto.txt")
    resultado = {}
    for modelo in modelos:
        print(f"  perplexidade de {modelo} ...", file=sys.stderr)
        try:
            gguf = Path(ggufs[modelo]).expanduser() if modelo in ggufs else gguf_do_ollama(modelo)
            resultado[modelo] = round(medir_perplexidade(binario, gguf, texto), 3)
        except qa.ErroUsuario as erro:
            # Uma falha aqui não invalida as outras métricas: avisa e segue.
            print(f"aviso: {erro}", file=sys.stderr)
            resultado[modelo] = None
    return resultado


# ======================================================================================================================
# EXECUÇÃO
# ======================================================================================================================


def avaliar(
    gabarito: list[dict],
    indice: qa.Indice,
    modos: list[str],
    modelos: list[str],
    pasta: Path,
    avaliador=None,
) -> list[dict]:
    """Roda todas as perguntas em todos os modos e modelos e calcula as métricas.

    Entradas:
        gabarito: saída de ``carregar_gabarito``.
        indice: índice do manual.
        modos: lista entre S0, S1, S2 e S3.
        modelos: modelos leitores a comparar.
        pasta: onde gravar respostas.jsonl.
        avaliador: BERTScore carregado (None = sem BERTScore).
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
                    print(
                        f"  [{feitos}/{total}] {nome_modo} {modelo} pergunta {item['id']}",
                        file=sys.stderr,
                    )
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
                # BERTScore em lote para as respostas deste modo e modelo.
                calcular_bertscore(registros, avaliador)
                # Grava as respostas deste modo e modelo (com o BERTScore de cada uma).
                for registro in registros:
                    arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
                arquivo.flush()
                linhas.append({"modo": nome_modo, "modelo": modelo, **resumir(registros)})
    return linhas


def gravar_metricas(linhas: list[dict], pasta: Path) -> None:
    """Grava metrics.csv e, se o matplotlib existir, os gráficos.

    grafico.png tem EM, F1, BERTScore e Recall@5 por modo e modelo; grafico_perplexidade.png, a perplexidade de cada
    modelo (só se ela foi calculada).

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
    # Um grupo de barras por linha (modo + modelo), com EM, F1, BERTScore e Recall@5.
    rotulos = [f"{linha['modo']}\n{linha['modelo']}" for linha in linhas]
    metricas = ["em", "f1", "bertscore_f1", "recall_at_5"]
    largura = 0.2
    figura, eixo = plt.subplots(figsize=(max(6, 1.4 * len(linhas)), 4))
    for n, metrica in enumerate(metricas):
        valores = [linha[metrica] or 0 for linha in linhas]
        eixo.bar(
            [i + (n - 1.5) * largura for i in range(len(linhas))],
            valores,
            width=largura,
            label=metrica,
        )
    eixo.set_xticks(range(len(linhas)), rotulos)
    eixo.set_ylim(0, 1)
    eixo.legend()
    eixo.set_title("Métricas por modo de busca")
    figura.tight_layout()
    figura.savefig(pasta / "grafico.png", dpi=150)
    plt.close(figura)
    # Perplexidade em gráfico separado: não fica entre 0 e 1 e é uma por modelo (não por modo).
    perplexidades = {linha["modelo"]: linha.get("perplexidade") for linha in linhas}
    perplexidades = {modelo: valor for modelo, valor in perplexidades.items() if valor is not None}
    if perplexidades:
        figura, eixo = plt.subplots(figsize=(max(4, 1.4 * len(perplexidades)), 4))
        eixo.bar(list(perplexidades), list(perplexidades.values()))
        eixo.set_ylabel("perplexidade (menor = melhor)")
        eixo.set_title("Perplexidade sobre o texto do manual")
        figura.tight_layout()
        figura.savefig(pasta / "grafico_perplexidade.png", dpi=150)
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
    parser.add_argument(
        "--modos",
        nargs="+",
        choices=qa.MODOS,
        default=list(qa.MODOS),
        help="padrão: S0 S1 S2 S3",
    )
    parser.add_argument("--modelos", nargs="+", default=[qa.MODELO_LEITOR], help="leitores a comparar")
    parser.add_argument("--limite", type=int, help="usa só as N primeiras perguntas aprovadas")
    parser.add_argument(
        "--sem-bertscore",
        action="store_true",
        help="não calcula o BERTScore (mais rápido)",
    )
    parser.add_argument("--sem-perplexidade", action="store_true", help="não calcula a perplexidade")
    parser.add_argument(
        "--llama-perplexity",
        default=str(LLAMA_PERPLEXITY),
        help=f"executável do llama.cpp (padrão: {LLAMA_PERPLEXITY})",
    )
    parser.add_argument(
        "--gguf",
        nargs="+",
        default=[],
        metavar="MODELO=ARQUIVO",
        help="GGUF de um modelo, se não vier do Ollama",
    )
    args = parser.parse_args(argv)
    try:
        indice = qa.carregar_indice()
        gabarito = carregar_gabarito(Path(args.respostas), indice)[: args.limite]
        # Cada execução ganha uma pasta própria com data e hora.
        pasta = qa.RAIZ / "resultados" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pasta.mkdir(parents=True)
        print(f"Avaliando {len(gabarito)} perguntas aprovadas ...", file=sys.stderr)
        # Carrega o BERTScore antes de começar (o download do BERTimbau acontece só na primeira vez).
        avaliador = None if args.sem_bertscore else criar_bertscore()
        linhas = avaliar(gabarito, indice, args.modos, args.modelos, pasta, avaliador)
        # Perplexidade: uma por modelo, repetida em todas as linhas (modos) desse modelo.
        if not args.sem_perplexidade:
            ggufs = dict(item.split("=", 1) for item in args.gguf)
            binario = Path(args.llama_perplexity).expanduser()
            perplexidades = calcular_perplexidades(args.modelos, indice, binario, ggufs, pasta)
            for linha in linhas:
                linha["perplexidade"] = perplexidades.get(linha["modelo"])
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
