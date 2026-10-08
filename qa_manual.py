#!/usr/bin/env python3
"""Perguntas e respostas sobre o manual técnico (RAG local com Ollama).

Trabalho da disciplina IC0024 (PGCOMP/UFBA). O sistema lê o ``manual.pdf`` que fica ao lado deste arquivo, divide o
texto em trechos, busca os trechos mais parecidos com cada pergunta e pede ao modelo de linguagem que responda usando
só esses trechos, citando a seção e a página. Tudo roda na própria máquina: o único serviço usado é o Ollama local.

Fluxo (Retriever-Reader):
    1. Leitura do PDF ........... texto de cada página, sem cabeçalho/rodapé repetido.
    2. Trechos .................. parágrafos agrupados por seção, até ~400 tokens, com seção e página.
    3. Índices ................. BM25 (palavras) e embeddings bge-m3 (significado), gravados em indice/.
    4. Busca ................... as duas listas de candidatos são fundidas por RRF; ficam os 5 melhores trechos.
    5. Resposta ................ qwen3:4b responde só com os trechos e termina com "(seção X, p. N)".

Comandos:
    python qa_manual.py indexar      # lê o manual.pdf e cria o índice (uma vez por versão do manual)
    python qa_manual.py perguntar    # modo interativo: Pergunta: / Resposta:
    python qa_manual.py responder    # perguntas.json -> resposta.json (respostas propostas para revisão)

As métricas ficam no arquivo separado ``avaliar.py``, que usa as funções daqui.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# ======================================================================================================================
# CONFIGURAÇÃO
# ======================================================================================================================

# Pasta onde está este arquivo: o manual, as perguntas e os resultados ficam todos aqui.
RAIZ = Path(__file__).resolve().parent
# Arquivos de entrada e saída padrão.
ARQ_MANUAL = RAIZ / "manual.pdf"
ARQ_PERGUNTAS = RAIZ / "perguntas.json"
ARQ_RESPOSTAS = RAIZ / "resposta.json"
# Pasta do índice: trechos (texto + seção + página) e vetores dos embeddings.
PASTA_INDICE = RAIZ / "indice"
ARQ_TRECHOS = PASTA_INDICE / "trechos.json"
ARQ_VETORES = PASTA_INDICE / "vetores.npy"

# Ollama local: única rede usada pelo sistema (o manual nunca sai da máquina).
OLLAMA_HOST = "http://localhost:11434"
# Modelo que lê os trechos e escreve a resposta.
MODELO_LEITOR = "qwen3:4b"
# Modelo que transforma textos em vetores (busca por significado).
MODELO_EMBEDDINGS = "bge-m3"
# Modelo que escreve as respostas PROPOSTAS do comando responder. É diferente do leitor de propósito: a resposta
# aprovada vira gabarito na avaliação, e um leitor não deve ser comparado com respostas escritas por ele mesmo.
MODELO_GERADOR = "llama3.2:3b"

# Tamanho máximo de um trecho, estimado em tokens (palavras x 1,3 é uma boa aproximação para o português).
MAX_TOKENS_TRECHO = 400
TOKENS_POR_PALAVRA = 1.3
# Blocos com menos caracteres que isso (números de página soltos, fragmentos) são descartados.
MIN_CARACTERES_PARAGRAFO = 25
# Quantos trechos vão para o leitor, quantos candidatos cada busca traz antes da fusão e a constante do RRF.
K = 5
K_CANDIDATOS = 20
RRF_C = 60
# Decodificação determinística: mesma pergunta, mesma resposta.
OPCOES_GERACAO = {"temperature": 0, "top_p": 1.0, "seed": 42, "num_ctx": 8192, "num_predict": 200}
# Frase que o leitor usa quando a resposta não está nos trechos.
FRASE_ABSTENCAO = "Não encontrado no manual"

# Título de seção: número (1, 2.1, 4.2.3...) seguido de texto começando em maiúscula. Ex.: "4.2 Liberação de carga".
REGEX_TITULO = re.compile(r"^(\d+(?:\.\d+){0,3})\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ].{2,80})$")
# Linha de sumário ("4.2 Liberação de carga ........ 17"): não é título nem conteúdo.
REGEX_SUMARIO = re.compile(r"(\.\s?){4,}\s*\d+\s*$")

# Palavras muito comuns que não ajudam a busca por palavras (BM25).
STOPWORDS = set(
    "a ao aos as até com como da das de dela dele deles do dos e ela elas ele eles em entre era essa esse esta este "
    "eu foi for foram há isso isto já lhe mais mas me mesmo muito na nas nem no nos não o os ou para pela pelas pelo "
    "pelos por qual quando que quem se sem ser seu seus sua suas são só também tem um uma você à às é".split()
)

# Modos de busca usados nos experimentos: S0 (sem busca), S1 (BM25), S2 (denso), S3 (híbrido).
MODOS = {"S0": "nenhum", "S1": "bm25", "S2": "denso", "S3": "hibrido"}

# Prompt do leitor com trechos (as chaves duplas viram chaves simples no texto final).
PROMPT_LEITOR = """Você é um assistente que responde perguntas sobre um manual técnico do setor portuário.
Use APENAS os trechos abaixo. Se a informação não estiver neles, responda exatamente:
Não encontrado no manual.
Responda em uma ou duas frases, em português, e termine indicando a fonte no formato:
(seção {{tema_id}}, p. {{pagina}})

Trechos:
{trechos}

Pergunta: {pergunta}
Resposta:"""

# Prompt do leitor sem trechos (modo S0, mede o que o modelo sabe sozinho).
PROMPT_SEM_TRECHOS = """Você é um assistente que responde perguntas sobre um manual técnico do setor portuário.
Responda em uma ou duas frases, em português. Se não souber, responda exatamente:
Não encontrado no manual.

Pergunta: {pergunta}
Resposta:"""


class ErroUsuario(Exception):
    """Erro com mensagem que diz ao usuário o que fazer (o programa imprime e sai com código 2)."""


# ======================================================================================================================
# OLLAMA
# ======================================================================================================================


def criar_cliente():
    """Cria o cliente do Ollama local.

    Entradas:
        nenhuma (usa ``OLLAMA_HOST``).
    Saídas:
        ``ollama.Client`` apontando para localhost, sem proxy (``trust_env=False``), para garantir que nenhum texto do
        manual seja enviado a outro endereço.
    Erros:
        ErroUsuario se o pacote ``ollama`` não estiver instalado.
    """
    # Importa aqui para o --help funcionar mesmo sem o pacote instalado.
    try:
        import ollama
    except ImportError:
        raise ErroUsuario("pacote 'ollama' não instalado; rode: pip install ollama") from None
    # Cliente local, com tempo limite generoso (modelos em CPU são lentos).
    return ollama.Client(host=OLLAMA_HOST, timeout=300, trust_env=False)


def _chamar_ollama(funcao, modelo: str):
    """Executa uma chamada ao Ollama traduzindo os erros comuns em mensagens claras.

    Entradas:
        funcao: função sem argumentos que faz a chamada.
        modelo: nome do modelo usado (para a mensagem de modelo ausente).
    Saídas:
        o que a chamada devolver.
    Erros:
        ErroUsuario se o Ollama estiver desligado ou o modelo não estiver baixado.
    """
    import ollama

    try:
        return funcao()
    except ConnectionError:
        # O pacote ollama converte falha de conexão em ConnectionError.
        raise ErroUsuario(f"Ollama não respondeu em {OLLAMA_HOST}; abra o Ollama ou rode `ollama serve`") from None
    except ollama.ResponseError as erro:
        # 404 = modelo não baixado.
        if erro.status_code == 404:
            raise ErroUsuario(f"modelo {modelo} não encontrado; rode `ollama pull {modelo}`") from None
        raise


def gerar_embeddings(cliente, textos: list[str]):
    """Transforma textos em vetores normalizados (passo 3, parte densa).

    Entradas:
        cliente: cliente do Ollama.
        textos: lista de textos.
    Saídas:
        matriz numpy (len(textos) x 1024) com vetores de norma 1, para que o produto interno seja o cosseno.
    Erros:
        ErroUsuario (via ``_chamar_ollama``).
    """
    import numpy as np

    vetores = []
    # Envia em lotes de 16 para não estourar a memória do Ollama em CPU.
    for inicio in range(0, len(textos), 16):
        lote = textos[inicio : inicio + 16]
        resposta = _chamar_ollama(
            lambda lote=lote: cliente.embed(model=MODELO_EMBEDDINGS, input=lote), MODELO_EMBEDDINGS
        )
        vetores.extend(resposta.embeddings)
    matriz = np.asarray(vetores, dtype=np.float32)
    # Normaliza cada linha (evita divisão por zero em vetores nulos).
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


def conversar(cliente, modelo: str, prompt: str) -> str:
    """Pede uma resposta ao modelo de linguagem.

    Entradas:
        cliente: cliente do Ollama.
        modelo: nome do modelo (ex.: qwen3:4b).
        prompt: texto completo enviado ao modelo.
    Saídas:
        texto da resposta, sem o bloco de raciocínio <think> que o Qwen3 às vezes devolve.
    Erros:
        ErroUsuario (via ``_chamar_ollama``).
    """
    mensagens = [{"role": "user", "content": prompt}]

    # Pede a resposta sem o modo de raciocínio (mais rápido e direto).
    def chamada():
        try:
            return cliente.chat(model=modelo, messages=mensagens, options=OPCOES_GERACAO, think=False)
        except Exception as erro:
            # Modelos sem suporte a raciocínio (ex.: llama3.2) recusam o parâmetro think: refaz sem ele.
            if "think" in str(erro).lower():
                return cliente.chat(model=modelo, messages=mensagens, options=OPCOES_GERACAO)
            raise

    resposta = _chamar_ollama(chamada, modelo)
    texto = resposta.message.content or ""
    # Remove qualquer bloco <think>...</think> que tenha vindo mesmo assim.
    texto = re.sub(r"<think>.*?(</think>|$)", "", texto, flags=re.DOTALL)
    return texto.strip()


# ======================================================================================================================
# PASSOS 1 E 2 — LEITURA DO PDF E TRECHOS
# ======================================================================================================================


def ler_blocos(caminho_pdf: Path) -> list[tuple[int, str]]:
    """Lê os blocos de texto do PDF, página por página, sem cabeçalhos e rodapés repetidos (passo 1).

    Entradas:
        caminho_pdf: caminho do manual em PDF.
    Saídas:
        lista de (número da página, texto do bloco), na ordem de leitura.
    Erros:
        ErroUsuario se o PDF não abrir ou não tiver texto selecionável (PDF escaneado).
    """
    try:
        import pymupdf
    except ImportError:
        raise ErroUsuario("pacote 'pymupdf' não instalado; rode: pip install pymupdf") from None

    blocos = []
    try:
        documento = pymupdf.open(caminho_pdf)
    except Exception:
        raise ErroUsuario(f"não foi possível abrir o PDF {caminho_pdf}") from None
    with documento:
        n_paginas = documento.page_count
        for numero, pagina in enumerate(documento, start=1):
            # Cada bloco é (x0, y0, x1, y1, texto, nº do bloco, tipo); tipo 0 = texto, 1 = imagem.
            textos = [b for b in pagina.get_text("blocks") if b[6] == 0 and b[4].strip()]
            # Ordena de cima para baixo e da esquerda para a direita (ordem de leitura).
            textos.sort(key=lambda b: (round(b[1]), b[0]))
            blocos.extend((numero, b[4]) for b in textos)

    # PDF escaneado não tem camada de texto: menos de 50 caracteres por página em média.
    if n_paginas == 0 or sum(len(t) for _, t in blocos) < 50 * n_paginas:
        raise ErroUsuario("PDF sem texto selecionável; rode OCR antes (ex.: ocrmypdf manual.pdf manual_ocr.pdf)")

    # Cabeçalho/rodapé: o mesmo texto (ignorando números) em mais da metade das páginas.
    def chave(texto: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"\d", "", texto)).strip().lower()

    paginas_por_chave: dict[str, set[int]] = {}
    for numero, texto in blocos:
        paginas_por_chave.setdefault(chave(texto), set()).add(numero)
    repetidos = {c for c, pags in paginas_por_chave.items() if len(pags) >= 2 and len(pags) > 0.5 * n_paginas}
    # Devolve só os blocos que não são cabeçalho nem rodapé.
    return [(numero, texto) for numero, texto in blocos if chave(texto) not in repetidos]


def limpar(texto: str) -> str:
    """Limpa o texto de um bloco.

    Entradas:
        texto: texto bruto do bloco.
    Saídas:
        texto em uma linha: hifenização de fim de linha desfeita ("libera-\\nção" -> "liberação"), quebras de
        linha trocadas por espaço e espaços repetidos colapsados.
    """
    # Normaliza acentos para a forma composta (á em vez de a + ´).
    texto = unicodedata.normalize("NFC", texto)
    # Junta palavra quebrada com hífen no fim da linha quando a linha seguinte começa em minúscula.
    texto = re.sub(r"(\w)-\s*\n\s*(?=[a-zà-ÿ])", r"\1", texto)
    # Troca quebras de linha e espaços repetidos por um único espaço.
    return re.sub(r"\s+", " ", texto).strip()


def numero_plausivel(novo: str, atual: str) -> bool:
    """Confere se a numeração de um título pode vir depois da seção atual.

    Evita que passos numerados ("1 Acione o alarme") ou linhas de tabela ("202 Armazém 3") virem seções.

    Entradas:
        novo: numeração do candidato a título (ex.: "2.4").
        atual: numeração da seção atual ("" antes do primeiro título).
    Saídas:
        True se o candidato avança até 5 posições em algum nível (2.3 -> 2.4, 3, 3.1) ou abre subseção (2.3 -> 2.3.1).
    """
    # O primeiro título do documento é sempre aceito.
    if not atual:
        return True
    n = [int(x) for x in novo.split(".")]
    a = [int(x) for x in atual.split(".")]
    # Conta quantos níveis iniciais são iguais (ex.: 2.3 e 2.4 têm 1 nível em comum).
    comum = 0
    while comum < min(len(n), len(a)) and n[comum] == a[comum]:
        comum += 1
    # Repetir a seção atual ou voltar para um nível acima dela não é plausível.
    if comum == len(n):
        return False
    # No nível em que diferem, o número tem de avançar de 1 a 5.
    if comum < len(a) and not 0 < n[comum] - a[comum] <= 5:
        return False
    # Níveis novos abaixo do ponto de avanço começam baixos (1 a 5).
    inicio = comum + 1 if comum < len(a) else comum
    return all(1 <= x <= 5 for x in n[inicio:])


def criar_trechos(caminho_pdf: Path) -> list[dict]:
    """Transforma o PDF em trechos com seção e página (passos 1 e 2).

    Entradas:
        caminho_pdf: caminho do manual.
    Saídas:
        lista de trechos ``{"id", "secao", "secao_id", "pagina_inicio", "pagina_fim", "texto"}``. Cada trecho junta
        parágrafos inteiros da mesma seção até ~400 tokens; um parágrafo maior que isso vira um trecho sozinho.
    Erros:
        ErroUsuario se o PDF não tiver texto ou não gerar nenhum parágrafo.
    """
    # Passo 1: blocos limpos, cada um com a sua página.
    blocos = ler_blocos(caminho_pdf)

    # Passo 2a: percorre os blocos montando parágrafos e acompanhando a seção corrente.
    paragrafos = []
    secao, secao_id = "Sem título", ""
    for pagina, bruto in blocos:
        # Descarta linhas de sumário e linhas vazias.
        linhas = [linha.strip() for linha in bruto.splitlines() if linha.strip() and not REGEX_SUMARIO.search(linha)]
        if not linhas:
            continue
        # Se a primeira linha é um título numerado em sequência, ela abre uma nova seção.
        titulo = REGEX_TITULO.match(limpar(linhas[0]))
        if titulo and numero_plausivel(titulo.group(1), secao_id):
            secao, secao_id = limpar(linhas[0]), titulo.group(1)
            linhas = linhas[1:]
        # O resto do bloco é um parágrafo da seção corrente (se não for curto demais).
        texto = limpar("\n".join(linhas))
        if len(texto) >= MIN_CARACTERES_PARAGRAFO:
            paragrafos.append({"secao": secao, "secao_id": secao_id, "pagina": pagina, "texto": texto})
    if not paragrafos:
        raise ErroUsuario(f"nenhum parágrafo extraído de {caminho_pdf}; confira se o PDF tem texto")

    # Passo 2b: agrupa parágrafos seguidos da mesma seção até o limite de tokens.
    def tokens(texto: str) -> int:
        return round(len(texto.split()) * TOKENS_POR_PALAVRA)

    grupos: list[list[dict]] = []
    for paragrafo in paragrafos:
        ultimo = grupos[-1] if grupos else None
        # Começa um grupo novo se mudou a seção ou se o parágrafo não cabe no grupo atual.
        mesma_secao = ultimo and ultimo[-1]["secao"] == paragrafo["secao"]
        cabe = ultimo and sum(tokens(p["texto"]) for p in ultimo) + tokens(paragrafo["texto"]) <= MAX_TOKENS_TRECHO
        if mesma_secao and cabe:
            ultimo.append(paragrafo)
        else:
            grupos.append([paragrafo])

    # Converte cada grupo em um trecho com id sequencial (t0001, t0002...).
    return [
        {
            "id": f"t{numero:04d}",
            "secao": grupo[0]["secao"],
            "secao_id": grupo[0]["secao_id"],
            "pagina_inicio": grupo[0]["pagina"],
            "pagina_fim": grupo[-1]["pagina"],
            "texto": "\n".join(p["texto"] for p in grupo),
        }
        for numero, grupo in enumerate(grupos, start=1)
    ]


# ======================================================================================================================
# PASSO 3 — ÍNDICES (BM25 E EMBEDDINGS)
# ======================================================================================================================


def tokenizar(texto: str) -> list[str]:
    """Divide um texto em palavras para o BM25.

    Entradas:
        texto: qualquer texto (trecho ou pergunta).
    Saídas:
        palavras em minúsculas, sem pontuação, sem palavras de 1 letra e sem stopwords (acentos mantidos).
    """
    palavras = re.findall(r"\w+", unicodedata.normalize("NFC", texto.lower()))
    return [p for p in palavras if len(p) > 1 and p not in STOPWORDS]


class Indice:
    """Trechos do manual com os dois índices de busca carregados na memória.

    Campos:
        trechos: lista de trechos (ver ``criar_trechos``).
        vetores: matriz de embeddings, uma linha por trecho.
        bm25: índice BM25 montado sobre as palavras dos trechos.
    """

    def __init__(self, trechos: list[dict], vetores):
        """Monta o índice BM25 sobre os trechos e guarda os vetores.

        Entradas:
            trechos: lista de trechos.
            vetores: matriz numpy de embeddings (mesma ordem dos trechos).
        Erros:
            ErroUsuario se o pacote bm25s não estiver instalado.
        """
        try:
            import bm25s
        except ImportError:
            raise ErroUsuario("pacote 'bm25s' não instalado; rode: pip install bm25s") from None
        self.trechos = trechos
        self.vetores = vetores
        # Mapa id -> trecho, usado para achar um trecho pelo id.
        self.por_id = {t["id"]: t for t in trechos}
        # Índice BM25 (k1 = 1,5 e b = 0,75); trecho sem palavras recebe um marcador para não ficar vazio.
        self.bm25 = bm25s.BM25(k1=1.5, b=0.75)
        self.bm25.index([tokenizar(t["texto"]) or ["_vazio_"] for t in trechos], show_progress=False)


def indexar(caminho_pdf: Path) -> int:
    """Cria o índice do manual e grava em ``indice/`` (passos 1 a 3).

    Entradas:
        caminho_pdf: caminho do manual.
    Saídas:
        número de trechos criados. Grava ``indice/trechos.json`` e ``indice/vetores.npy``.
    Erros:
        ErroUsuario se o PDF tiver problema ou o Ollama não estiver disponível.
    """
    import numpy as np

    # Passos 1 e 2: PDF -> trechos.
    trechos = criar_trechos(caminho_pdf)
    # Passo 3: embeddings de todos os trechos (é a parte demorada).
    vetores = gerar_embeddings(criar_cliente(), [t["texto"] for t in trechos])
    # Grava os dois arquivos do índice.
    PASTA_INDICE.mkdir(exist_ok=True)
    ARQ_TRECHOS.write_text(json.dumps(trechos, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(ARQ_VETORES, vetores)
    return len(trechos)


def carregar_indice() -> Indice:
    """Lê o índice gravado pelo comando ``indexar``.

    Saídas:
        objeto ``Indice`` pronto para buscas.
    Erros:
        ErroUsuario se o índice não existir.
    """
    import numpy as np

    if not ARQ_TRECHOS.exists() or not ARQ_VETORES.exists():
        raise ErroUsuario("índice não encontrado; rode antes: python qa_manual.py indexar")
    trechos = json.loads(ARQ_TRECHOS.read_text(encoding="utf-8"))
    return Indice(trechos, np.load(ARQ_VETORES))


# ======================================================================================================================
# PASSO 4 — BUSCA (BM25, DENSA E HÍBRIDA COM RRF)
# ======================================================================================================================


def buscar_bm25(indice: Indice, pergunta: str, k: int) -> list[str]:
    """Busca por palavras em comum (BM25).

    Entradas:
        indice: índice carregado.
        pergunta: texto da pergunta.
        k: quantos trechos devolver.
    Saídas:
        ids dos trechos em ordem de relevância (só os que têm alguma palavra em comum).
    """
    palavras = tokenizar(pergunta)
    if not palavras:
        return []
    # retrieve devolve as posições dos documentos e os scores, para uma lista de consultas.
    posicoes, scores = indice.bm25.retrieve([palavras], k=min(k, len(indice.trechos)), show_progress=False)
    return [indice.trechos[int(p)]["id"] for p, s in zip(posicoes[0], scores[0], strict=True) if s > 0]


def buscar_denso(indice: Indice, cliente, pergunta: str, k: int) -> list[str]:
    """Busca por significado (cosseno entre o embedding da pergunta e os dos trechos).

    Entradas:
        indice: índice carregado.
        cliente: cliente do Ollama (para o embedding da pergunta).
        pergunta: texto da pergunta.
        k: quantos trechos devolver.
    Saídas:
        ids dos trechos em ordem de similaridade.
    """
    vetor = gerar_embeddings(cliente, [pergunta])[0]
    # Como os vetores têm norma 1, o produto interno é o cosseno.
    similaridades = indice.vetores @ vetor
    melhores = similaridades.argsort()[::-1][:k]
    return [indice.trechos[int(i)]["id"] for i in melhores]


def fundir_rrf(listas: list[list[str]], k: int) -> list[str]:
    """Funde listas de resultados por Reciprocal Rank Fusion.

    Entradas:
        listas: listas de ids ordenadas por relevância.
        k: tamanho da lista final.
    Saídas:
        ids ordenados pela soma de 1 / (60 + posição) em cada lista; empates desfeitos pelo id.
    """
    pontos: Counter[str] = Counter()
    for lista in listas:
        for posicao, id_trecho in enumerate(lista, start=1):
            pontos[id_trecho] += 1.0 / (RRF_C + posicao)
    return [i for i, _ in sorted(pontos.items(), key=lambda par: (-par[1], par[0]))][:k]


def buscar(indice: Indice, cliente, pergunta: str, modo: str = "hibrido", k: int = K) -> list[str]:
    """Escolhe os trechos que vão para o leitor (passo 4).

    Entradas:
        indice: índice carregado (pode ser None no modo "nenhum").
        cliente: cliente do Ollama.
        pergunta: texto da pergunta.
        modo: "nenhum" (S0), "bm25" (S1), "denso" (S2) ou "hibrido" (S3).
        k: quantos trechos devolver.
    Saídas:
        ids dos trechos escolhidos (lista vazia no modo "nenhum").
    """
    if modo == "nenhum":
        return []
    if modo == "bm25":
        return buscar_bm25(indice, pergunta, k)
    if modo == "denso":
        return buscar_denso(indice, cliente, pergunta, k)
    # Híbrido: 20 candidatos de cada busca, fundidos por RRF.
    listas = [buscar_bm25(indice, pergunta, K_CANDIDATOS), buscar_denso(indice, cliente, pergunta, K_CANDIDATOS)]
    return fundir_rrf(listas, k)


# ======================================================================================================================
# PASSO 5 — RESPOSTA E VERIFICAÇÃO DA CITAÇÃO
# ======================================================================================================================


def montar_prompt(pergunta: str, trechos: list[dict]) -> str:
    """Monta o texto enviado ao leitor.

    Entradas:
        pergunta: texto da pergunta.
        trechos: trechos escolhidos pela busca (vazio no modo S0).
    Saídas:
        prompt com os trechos numerados no formato "[n] (seção X, p. N) texto".
    """
    if not trechos:
        return PROMPT_SEM_TRECHOS.format(pergunta=pergunta.strip())
    blocos = [
        f"[{n}] (seção {t['secao_id'] or t['secao']}, p. {t['pagina_inicio']}) {t['texto']}"
        for n, t in enumerate(trechos, start=1)
    ]
    return PROMPT_LEITOR.format(trechos="\n\n".join(blocos), pergunta=pergunta.strip())


def extrair_citacoes(resposta: str) -> list[tuple[str | None, int | None]]:
    """Encontra as citações de fonte na resposta.

    Entradas:
        resposta: texto da resposta do modelo.
    Saídas:
        lista de (seção, página), por exemplo [("4.2", 17)]; aceita "(seção 4.2, p. 17)", "seção 4.2", "p. 17" e
        "página 17". O campo que faltar fica None.
    """
    citacoes = []
    # Forma completa: seção e página juntas.
    for m in re.finditer(
        r"se[çc][ãa]o\s*(\d+(?:\.\d+)*)\s*[,;-]?\s*(?:p\.|p[áa]g\.?|p[áa]gina)\s*(\d+)", resposta, re.I
    ):
        citacoes.append((m.group(1), int(m.group(2))))
    if citacoes:
        return citacoes
    # Formas parciais: só a seção ou só a página.
    for m in re.finditer(r"se[çc][ãa]o\s*(\d+(?:\.\d+)*)", resposta, re.I):
        citacoes.append((m.group(1), None))
    for m in re.finditer(r"(?<![\w.])(?:p\.|p[áa]g\.?|p[áa]gina)\s*(\d+)", resposta, re.I):
        citacoes.append((None, int(m.group(1))))
    return citacoes


def citacao_valida(citacoes: list[tuple[str | None, int | None]], trechos: list[dict]) -> bool:
    """Confere se alguma citação aponta para um dos trechos usados.

    Entradas:
        citacoes: saída de ``extrair_citacoes``.
        trechos: trechos que o leitor recebeu.
    Saídas:
        True se seção e página citadas batem com algum trecho (a página dentro do intervalo do trecho).
    """
    for secao, pagina in citacoes:
        for t in trechos:
            secao_ok = secao is None or secao == t["secao_id"]
            pagina_ok = pagina is None or t["pagina_inicio"] <= pagina <= t["pagina_fim"]
            if (secao or pagina) and secao_ok and pagina_ok:
                return True
    return False


def absteve(resposta: str) -> bool:
    """Diz se a resposta é a frase de abstenção ("Não encontrado no manual"), ignorando acentos e maiúsculas.

    Entradas:
        resposta: texto da resposta.
    Saídas:
        True se a resposta contém a frase de abstenção.
    """

    def simplificar(texto: str) -> str:
        sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
        return re.sub(r"\W+", " ", sem_acento).strip()

    return simplificar(FRASE_ABSTENCAO) in simplificar(resposta)


def responder(
    pergunta: str, indice: Indice | None, cliente, modo: str = "hibrido", modelo: str = MODELO_LEITOR
) -> dict:
    """Responde uma pergunta: busca os trechos, gera a resposta e confere a citação (passos 4 e 5).

    Entradas:
        pergunta: texto da pergunta.
        indice: índice carregado (None só no modo "nenhum").
        cliente: cliente do Ollama.
        modo: modo de busca ("nenhum", "bm25", "denso" ou "hibrido").
        modelo: modelo que escreve a resposta.
    Saídas:
        dicionário com ``resposta``, ``trechos`` (ids na ordem da busca), ``absteve``, ``citacoes``,
        ``citacao_valida`` e ``latencia_s``.
    """
    inicio = time.perf_counter()
    # Passo 4: busca.
    ids = buscar(indice, cliente, pergunta, modo)
    trechos = [indice.por_id[i] for i in ids] if ids else []
    # Passo 5: resposta do leitor.
    resposta = conversar(cliente, modelo, montar_prompt(pergunta, trechos))
    citacoes = extrair_citacoes(resposta)
    return {
        "resposta": resposta,
        "trechos": ids,
        "absteve": absteve(resposta),
        "citacoes": citacoes,
        "citacao_valida": citacao_valida(citacoes, trechos),
        "latencia_s": round(time.perf_counter() - inicio, 2),
    }


# ======================================================================================================================
# COMANDO RESPONDER — perguntas.json -> resposta.json
# ======================================================================================================================


def ler_perguntas(caminho: Path) -> list[dict]:
    """Lê e valida o perguntas.json.

    Entradas:
        caminho: arquivo no formato [{"id": 1, "pergunta": "..."}, ...].
    Saídas:
        lista de {"id": texto, "pergunta": texto}.
    Erros:
        ErroUsuario se o arquivo não existir, não for JSON válido, ou tiver ids repetidos ou perguntas vazias.
    """
    if not caminho.exists():
        raise ErroUsuario(f"arquivo não encontrado: {caminho}")
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError as erro:
        raise ErroUsuario(f"{caminho.name} não é um JSON válido (linha {erro.lineno})") from None
    if not isinstance(dados, list) or not dados:
        raise ErroUsuario(f"{caminho.name} deve ser uma lista com pelo menos uma pergunta")
    perguntas, vistos = [], set()
    for posicao, item in enumerate(dados, start=1):
        # Cada item precisa de id e de uma pergunta não vazia.
        if not isinstance(item, dict) or "id" not in item or not str(item.get("pergunta", "")).strip():
            raise ErroUsuario(f"{caminho.name}: item {posicao} precisa de 'id' e 'pergunta'")
        id_ = str(item["id"])
        if id_ in vistos:
            raise ErroUsuario(f"{caminho.name}: id {id_} repetido")
        vistos.add(id_)
        perguntas.append({"id": id_, "pergunta": str(item["pergunta"]).strip()})
    return perguntas


def tipo_da_pergunta(pergunta: str, sem_resposta: bool) -> str:
    """Classifica a pergunta para a análise: sem_resposta, procedimental ou factual.

    Entradas:
        pergunta: texto da pergunta.
        sem_resposta: True se o sistema não encontrou a resposta.
    Saídas:
        "sem_resposta", "procedimental" (como fazer algo) ou "factual" (fato pontual).
    """
    if sem_resposta:
        return "sem_resposta"
    inicio = unicodedata.normalize("NFKD", pergunta.lower()).encode("ascii", "ignore").decode()
    procedimentos = (
        "como ",
        "de que forma",
        "quais os passos",
        "quais sao os passos",
        "qual o procedimento",
        "o que fazer",
    )
    return "procedimental" if inicio.startswith(procedimentos) else "factual"


def gerar_respostas_propostas(perguntas: list[dict], indice: Indice, cliente, modelo: str) -> list[dict]:
    """Escreve uma resposta proposta para cada pergunta, para revisão humana.

    Entradas:
        perguntas: saída de ``ler_perguntas``.
        indice: índice carregado.
        cliente: cliente do Ollama.
        modelo: modelo que escreve as respostas (padrão: o gerador, que não é o leitor avaliado).
    Saídas:
        lista no formato do resposta.json: id, tipo, pergunta, resposta (sem a citação), secao, pagina, trecho,
        aprovada (False) e observacao ("").
    """
    itens = []
    for numero, item in enumerate(perguntas, start=1):
        print(f"  [{numero}/{len(perguntas)}] pergunta {item['id']}", file=sys.stderr)
        resultado = responder(item["pergunta"], indice, cliente, "hibrido", modelo)
        trechos = [indice.por_id[i] for i in resultado["trechos"]]
        # De onde veio a resposta: o trecho citado, se a citação bater; senão, o trecho mais bem colocado.
        origem = None
        for secao, pagina in resultado["citacoes"]:
            origem = next((t for t in trechos if citacao_valida([(secao, pagina)], [t])), None)
            if origem:
                break
        origem = origem or (trechos[0] if trechos else None)
        sem_resposta = resultado["absteve"] or origem is None
        # A resposta vai sem a citação, que fica nos campos secao e pagina.
        texto = re.sub(r"\(?\s*se[çc][ãa]o[^()]*\)?\.?\s*$", "", resultado["resposta"], flags=re.I).strip()
        itens.append(
            {
                "id": item["id"],
                "tipo": tipo_da_pergunta(item["pergunta"], sem_resposta),
                "pergunta": item["pergunta"],
                "resposta": FRASE_ABSTENCAO if sem_resposta else texto,
                "secao": None if sem_resposta else origem["secao_id"],
                "pagina": None if sem_resposta else origem["pagina_inicio"],
                "trecho": None if sem_resposta else origem["id"],
                "aprovada": False,
                "observacao": "",
            }
        )
    return itens


# ======================================================================================================================
# LINHA DE COMANDO
# ======================================================================================================================


def comando_indexar(args) -> None:
    """Comando ``indexar``: lê o manual e cria o índice."""
    caminho = Path(args.manual)
    if not caminho.exists():
        raise ErroUsuario(f"manual não encontrado: {caminho} (coloque o manual.pdf ao lado do qa_manual.py)")
    print(f"Lendo {caminho} ...")
    inicio = time.perf_counter()
    n = indexar(caminho)
    print(f"{n} trechos indexados em {time.perf_counter() - inicio:.0f} s. Índice em {PASTA_INDICE}/")


def comando_perguntar(args) -> None:
    """Comando ``perguntar``: uma pergunta (--pergunta) ou o modo interativo Pergunta:/Resposta:."""
    # Carrega o índice e o cliente uma vez só, antes das perguntas.
    indice = carregar_indice()
    cliente = criar_cliente()
    modo = MODOS[args.modo]
    if args.pergunta:
        print(f"Resposta: {responder(args.pergunta, indice, cliente, modo, args.modelo)['resposta']}")
        return
    print("Digite a pergunta e tecle Enter. Linha vazia ou 'sair' encerra.\n")
    while True:
        try:
            pergunta = input("Pergunta:\n").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not pergunta or pergunta.lower() == "sair":
            break
        print(f"Resposta: {responder(pergunta, indice, cliente, modo, args.modelo)['resposta']}\n")
    print("Até logo.")


def comando_responder(args) -> None:
    """Comando ``responder``: lê perguntas.json e grava resposta.json para revisão."""
    saida = Path(args.saida)
    # Não apaga uma revisão já feita sem pedido explícito.
    if saida.exists() and not args.forcar:
        raise ErroUsuario(f"{saida.name} já existe (pode conter a sua revisão); use --forcar para sobrescrever")
    perguntas = ler_perguntas(Path(args.perguntas))
    indice = carregar_indice()
    print(f"Respondendo {len(perguntas)} perguntas com {args.modelo} ...", file=sys.stderr)
    itens = gerar_respostas_propostas(perguntas, indice, criar_cliente(), args.modelo)
    saida.write_text(json.dumps(itens, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGravado: {saida}")
    print('Revise cada resposta: marque "aprovada": true nas corretas (corrija o texto se preciso).')
    print("Depois rode: python avaliar.py")


def main(argv: list[str] | None = None) -> int:
    """Lê os argumentos da linha de comando e executa o comando pedido.

    Entradas:
        argv: argumentos (padrão: os do terminal).
    Saídas:
        código de saída: 0 sucesso, 2 erro de uso, 130 interrompido.
    """
    parser = argparse.ArgumentParser(prog="qa_manual.py", description="Perguntas e respostas sobre o manual.")
    comandos = parser.add_subparsers(dest="comando", required=True)

    p = comandos.add_parser("indexar", help="lê o manual.pdf e cria o índice")
    p.add_argument("--manual", default=str(ARQ_MANUAL), help="caminho do PDF (padrão: manual.pdf ao lado do .py)")
    p.set_defaults(funcao=comando_indexar)

    p = comandos.add_parser("perguntar", help="responde perguntas (modo interativo se não passar --pergunta)")
    p.add_argument("--pergunta", help="faz uma pergunta só, sem o modo interativo")
    p.add_argument("--modo", choices=MODOS, default="S3", help="S0 sem busca, S1 BM25, S2 denso, S3 híbrido")
    p.add_argument("--modelo", default=MODELO_LEITOR, help=f"modelo leitor (padrão: {MODELO_LEITOR})")
    p.set_defaults(funcao=comando_perguntar)

    p = comandos.add_parser("responder", help="perguntas.json -> resposta.json (para revisão)")
    p.add_argument("--perguntas", default=str(ARQ_PERGUNTAS))
    p.add_argument("--saida", default=str(ARQ_RESPOSTAS))
    p.add_argument(
        "--modelo", default=MODELO_GERADOR, help=f"modelo que propõe as respostas (padrão: {MODELO_GERADOR})"
    )
    p.add_argument("--forcar", action="store_true", help="sobrescreve o resposta.json existente")
    p.set_defaults(funcao=comando_responder)

    args = parser.parse_args(argv)
    try:
        args.funcao(args)
        return 0
    except ErroUsuario as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrompido.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
