#!/usr/bin/env python3
"""Perguntas e respostas sobre o manual técnico (RAG local com Ollama).

Trabalho da disciplina IC0024 (PGCOMP/UFBA). O sistema lê o ``manual.pdf`` que fica ao lado deste arquivo, divide o
texto em trechos, busca os trechos mais parecidos com cada pergunta e pede ao modelo de linguagem que responda usando
só esses trechos, citando a seção e a página. Tudo roda na própria máquina: o único serviço usado é o Ollama local.

Fluxo (Retriever-Reader), na mesma ordem em que as funções aparecem neste arquivo:
    0. Preparação .............. liga o Ollama local e cria o cliente.
    1. Leitura do PDF ........... texto e tabelas de cada página (find_tables), sem cabeçalho/rodapé repetido.
    2. Trechos .................. frases inteiras e linhas de tabela ("coluna: valor"), agrupadas por seção em trechos
                                  de até ~400 tokens, com seção e página.
    3. Índices ................. BM25 (palavras) e embeddings bge-m3 (significado), gravados em indice/.
    4. Busca ................... as duas listas de candidatos são fundidas por RRF; ficam os 5 melhores trechos.
    5. Resposta ................ Qwen3-4B responde só com os trechos e termina com "(seção X, p. N)".
    6. Gabarito ................ (comando responder) respostas propostas para revisão humana.
    7. Ajuste fino ............. (comando treinar) LoRA do leitor com o manual, descrito abaixo.
    8. Linha de comando ........ main(), no fim do arquivo, é onde a execução começa.

Ajuste fino (etapa 7, opcional, comando treinar): o leitor é treinado com LoRA sobre o texto do manual (Mac com
Apple Silicon, via MLX) e registrado no Ollama como ``qwen3-manual``:
    1. Dados ......... todos os trechos do índice (o manual inteiro) viram ``modelos/dados/train.jsonl``.
    2. Treino ........ LoRA (QLoRA em 4 bits) com o mlx-lm sobre a versão MLX do mesmo modelo do leitor.
    3. Adaptador ..... o adaptador treinado (poucos MB) é convertido para GGUF (conversor de LoRA do llama.cpp).
    4. Junção ........ o adaptador é somado ao GGUF que o Ollama já tem do leitor (llama-export-lora). Só as camadas
                       treinadas mudam (e saem em 16 bits); o resto é copiado como está.
    4b. Compactação .. as camadas treinadas voltam para o mesmo tipo de 4/6 bits do original (llama-quantize), e o
                       modelo ajustado fica do mesmo tamanho do original (~2,5 GB).
    5. Ollama ........ o resultado é registrado como ``qwen3-manual``, com o mesmo formato de conversa do leitor.
O pico de disco do treino é de ~6 GB (no passo 4); no fim sobram ~2,5 GB (o qwen3-manual no Ollama). Os pesos MLX
baixados para o treino (~2,3 GB) e os arquivos intermediários são apagados ao longo do caminho.

Comandos:
    python main.py                   # modo interativo: Pergunta: / Resposta:
    python main.py responder         # perguntas.json -> resposta.json (respostas propostas para revisão)
    python main.py treinar           # ajuste fino e registro do qwen3-manual (uma vez; de minutos a horas)
    python main.py indexar           # refaz o índice à força (normalmente não é preciso)

O sistema cuida sozinho do resto: liga o Ollama se ele estiver desligado (e desliga ao terminar), baixa os modelos
que faltarem e cria o índice na primeira vez (ou de novo quando o manual.pdf muda).

As métricas ficam no arquivo separado ``avaliador.py``, que usa as funções daqui.

Mapa do arquivo (cada função tem um cabeçalho com o número abaixo e quem a usa):
    Etapa 0 — Preparação: erros, conexão com o Ollama e barra de progresso
        0.1  ErroUsuario
        0.2  ollama_ligado
        0.3  ligar_ollama
        0.4  criar_cliente
        0.5  _chamar_ollama
        0.6  Progresso
    Etapa 1 — Leitura do PDF (texto e tabelas)
        1.1  limpar
        1.2  ler_tabela
        1.3  ler_elementos
    Etapa 2 — Divisão em parágrafos e trechos
        2.1  contar_tokens
        2.2  numero_plausivel
        2.3  juntar_tabelas_continuadas
        2.4  paragrafos_da_tabela
        2.5  criar_paragrafos
        2.6  dividir_paragrafo
        2.7  criar_trechos
    Etapa 3 — Indexação (BM25 e embeddings)
        3.1  tokenizar
        3.2  gerar_embeddings
        3.3  impressao_digital
        3.4  indexar
        3.5  Indice
        3.6  carregar_indice
    Etapa 4 — Busca (BM25, densa e híbrida com RRF)
        4.1  buscar_bm25
        4.2  buscar_denso
        4.3  fundir_rrf
        4.4  buscar
    Etapa 5 — Resposta e verificação da citação
        5.1  montar_prompt
        5.2  conversar
        5.3  extrair_citacoes
        5.4  citacao_valida
        5.5  absteve
        5.6  responder
    Etapa 6 — Respostas propostas para o gabarito (comando responder)
        6.1  ler_perguntas
        6.2  tipo_da_pergunta
        6.3  gerar_respostas_propostas
    Etapa 7 — Ajuste fino LoRA (comando treinar)
        7.1  rodar
        7.2  preparar_dados
        7.3  treinar
        7.4  converter_adaptador
        7.5  gguf_do_ollama
        7.6  juntar_no_leitor
        7.7  registrar_no_ollama
    Etapa 8 — Linha de comando (ponto de entrada)
        8.1  comando_indexar
        8.2  comando_perguntar
        8.3  comando_responder
        8.4  comando_treinar
        8.5  main
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path


# ======================================================================================================================
# CONFIGURAÇÃO (agrupada pela etapa que usa cada valor)
# ======================================================================================================================

# --- Arquivos do projeto ----------------------------------------------------------------------------------------------
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
# Impressão digital (SHA-256) do manual indexado: se o manual.pdf mudar, o índice é refeito.
ARQ_HASH = PASTA_INDICE / "manual.sha256"

# --- Ambiente: ferramentas e caches dentro do projeto -----------------------------------------------------------------
# Tudo o que o projeto baixa ou compila fica dentro desta pasta (no mesmo disco do projeto), e não na pasta do
# usuário: o Python e o llama.cpp em ferramentas/, e os caches (Hugging Face, PyTorch, Matplotlib, pip) em .cache/.
# As variáveis são fixadas aqui, antes de qualquer import do transformers, mlx-lm ou matplotlib (que as leem ao abrir).
PASTA_FERRAMENTAS = RAIZ / "ferramentas"
PASTA_CACHE = RAIZ / ".cache"
LLAMA_CPP = PASTA_FERRAMENTAS / "llama.cpp"
os.environ["HF_HOME"] = str(PASTA_CACHE / "huggingface")
os.environ["TORCH_HOME"] = str(PASTA_CACHE / "torch")
os.environ["MPLCONFIGDIR"] = str(PASTA_CACHE / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(PASTA_CACHE)
os.environ["PIP_CACHE_DIR"] = str(PASTA_CACHE / "pip")

# --- Etapa 0: Ollama e modelos ----------------------------------------------------------------------------------------
# Largura (em caracteres) da barra de progresso das tarefas longas (0.6).
LARGURA_BARRA = 20

# Ollama local: única rede usada pelo sistema (o manual nunca sai da máquina).
OLLAMA_HOST = "http://localhost:11434"
# Os comandos "ollama ..." chamados por este programa (e pelo avaliador.py) obedecem à variável de ambiente
# OLLAMA_HOST. Fixá-la aqui garante que eles também falem só com o Ollama local, mesmo que ela aponte para outro lugar.
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
# Modelo que lê os trechos e escreve a resposta: Qwen3-4B Instruct (versão 2507), em 4 bits. A tag "qwen3:4b" do
# Ollama hoje aponta para a versão Thinking, que pensa antes de responder e gastaria o limite de tokens nisso; a tag
# completa também fixa a versão exata, que o ajuste fino precisa conhecer.
MODELO_LEITOR = "qwen3:4b-instruct-2507-q4_K_M"
# Modelo que transforma textos em vetores (busca por significado).
MODELO_EMBEDDINGS = "bge-m3"
# Modelo que escreve as respostas PROPOSTAS do comando responder. É diferente do leitor de propósito: a resposta
# aprovada vira gabarito na avaliação, e um leitor não deve ser comparado com respostas escritas por ele mesmo.
MODELO_GERADOR = "llama3.2:3b"
# Nome, no Ollama, do leitor ajustado com o texto do manual pelo comando treinar (opcional).
MODELO_TREINADO = "qwen3-manual"

# --- Etapas 1 e 2: leitura do PDF e divisão em trechos ----------------------------------------------------------------
# Título de seção: número (1, 2.1, 4.2.3...), com ou sem ponto no fim, seguido de texto começando em maiúscula.
# Ex.: "1.4. BASE LEGAL E NORMATIVA" (no PDF o número fica numa linha e o título na de baixo; a etapa 1 junta as duas).
REGEX_TITULO = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{1,150})$")
# Linha de sumário ("4.2 Liberação de carga ........ 17"): não é título nem conteúdo.
REGEX_SUMARIO = re.compile(r"(\.\s?){4,}\s*\d+\s*$")
# Bloco só com numeração ("17", "1.4."): número de página ou resto do sumário, não é conteúdo.
REGEX_SO_NUMERO = re.compile(r"^[\d.\s]+$")
# Legenda de tabela ("Tabela 3 – TABELA DE NÃO CONFORMIDADES..."): dá nome às linhas da tabela seguinte.
REGEX_LEGENDA = re.compile(r"^Tabela\s+\d+")
# Tamanho máximo de um trecho, estimado em tokens. Medido com o tokenizador do Qwen3 sobre este manual, o português
# dá ~2,06 tokens por palavra (palavras longas, siglas e números). Com 5 trechos de 400 tokens, o prompt fica em
# ~2.300 tokens, bem abaixo do num_ctx (8192).
MAX_TOKENS_TRECHO = 400
TOKENS_POR_PALAVRA = 2.0
# Trecho menor que isso não fecha só porque a seção mudou: subseções de uma linha se juntam à seguinte.
MIN_TOKENS_TRECHO = 120
# Versão do jeito de dividir o manual: mudou a etapa 1 ou 2, aumenta aqui e o índice é refeito sozinho.
VERSAO_INDICE = 2

# --- Etapa 3: indexação -----------------------------------------------------------------------------------------------
# Palavras muito comuns que não ajudam a busca por palavras (BM25).
STOPWORDS = set(
    "a ao aos as até com como da das de dela dele deles do dos e ela elas ele eles em entre era essa esse esta este "
    "eu foi for foram há isso isto já lhe mais mas me mesmo muito na nas nem no nos não o os ou para pela pelas pelo "
    "pelos por qual quando que quem se sem ser seu seus sua suas são só também tem um uma você à às é".split()
)

# --- Etapa 4: busca ---------------------------------------------------------------------------------------------------
# Modos de busca usados nos experimentos: S0 (sem busca), S1 (BM25), S2 (denso), S3 (híbrido).
MODOS = {"S0": "nenhum", "S1": "bm25", "S2": "denso", "S3": "hibrido"}
# Quantos trechos vão para o leitor, quantos candidatos cada busca traz antes da fusão e a constante do RRF.
K = 5
K_CANDIDATOS = 20
RRF_C = 60

# --- Etapa 5: resposta ------------------------------------------------------------------------------------------------
# Decodificação determinística: mesma pergunta, mesma resposta. num_predict é o teto de tokens da resposta: as
# respostas do gabarito chegam a ~110 palavras (~230 tokens) mais a citação, e com 200 as mais longas eram cortadas
# antes da fonte. 400 dá folga sem deixar o modelo divagar (o prompt pede até três frases).
OPCOES_GERACAO = {"temperature": 0, "top_p": 1.0, "seed": 42, "num_ctx": 8192, "num_predict": 400}
# Frase que o leitor usa quando a resposta não está nos trechos.
FRASE_ABSTENCAO = "Não encontrado no manual"

# Prompt do leitor com trechos. Explica o formato das linhas de tabela, para o modelo ler "RIP: NÃO; Prazo: 0 dia"
# como colunas da mesma linha.
PROMPT_LEITOR = """Você é um assistente que responde perguntas sobre um manual técnico do setor portuário.
Use APENAS os trechos abaixo. Linhas de tabela aparecem como "[Tabela N, grupo] coluna: valor; coluna: valor".
Se a informação não estiver nos trechos, responda exatamente:
Não encontrado no manual.
Responda em português, de forma direta, em até três frases, e termine indicando a seção e a página do trecho
usado, no formato:
(seção X, p. N)

Trechos:
{trechos}

Pergunta: {pergunta}
Resposta:"""

# Prompt do leitor sem trechos (modo S0, mede o que o modelo sabe sozinho).
PROMPT_SEM_TRECHOS = """Você é um assistente que responde perguntas sobre um manual técnico do setor portuário.
Responda em português, de forma direta, em até três frases. Se não souber, responda exatamente:
Não encontrado no manual.

Pergunta: {pergunta}
Resposta:"""

# --- Etapa 7: ajuste fino ---------------------------------------------------------------------------------------------
# Ajuste fino. Versão MLX (4 bits) do mesmo modelo do leitor (Qwen3-4B Instruct 2507); o treino parte dela. Precisa
# ser a mesma versão do leitor do Ollama, porque o adaptador treinado é somado aos pesos dele.
MODELO_BASE_MLX = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
# Ferramentas do llama.cpp usadas no treino (conversor de LoRA, llama-export-lora e llama-quantize).
CONVERSOR_LORA = LLAMA_CPP / "convert_lora_to_gguf.py"
EXPORT_LORA = LLAMA_CPP / "build" / "bin" / "llama-export-lora"
QUANTIZE = LLAMA_CPP / "build" / "bin" / "llama-quantize"
# Semente do treino (mesma ordem dos exemplos a cada execução).
SEMENTE_TREINO = 42
# Hiperparâmetros do LoRA, pensados para caber em um Mac com 8 GB de memória.
EPOCAS = 2  # quantas vezes cada trecho de treino é visto
CAMADAS_LORA = 8  # quantas camadas (as últimas) recebem o adaptador
TAXA_APRENDIZADO = 1e-4
MAX_TOKENS_TREINO = 1024  # trechos maiores são cortados
# Arquivos e pastas do treino (todos dentro de modelos/, que não vai para o git).
PASTA_MODELOS = RAIZ / "modelos"
PASTA_DADOS = PASTA_MODELOS / "dados"
PASTA_ADAPTADOR = PASTA_MODELOS / "adaptador"
PASTA_PEFT = PASTA_MODELOS / "adaptador_peft"
ARQ_ADAPTADOR_GGUF = PASTA_MODELOS / "adaptador.gguf"
ARQ_JUNTO = PASTA_MODELOS / "junto-f16.gguf"
ARQ_TIPOS = PASTA_MODELOS / "tipos.txt"
ARQ_GGUF = PASTA_MODELOS / f"{MODELO_TREINADO}.gguf"
ARQ_MODELFILE = PASTA_MODELOS / "Modelfile"


# ======================================================================================================================
# ETAPA 0 — PREPARAÇÃO — ERROS, CONEXÃO COM O OLLAMA E BARRA DE PROGRESSO
#   Entrada: nada
#   Saída:   cliente do Ollama ligado em localhost (usado pelas etapas 3, 4 e 5) e a barra de progresso das
#            tarefas longas
#   Funções: 0.1 ErroUsuario · 0.2 ollama_ligado · 0.3 ligar_ollama · 0.4 criar_cliente · 0.5 _chamar_ollama
#            0.6 Progresso
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 0.1 ErroUsuario — erro com mensagem para o usuário; o programa imprime e sai com código 2
#     Usada por: todas as etapas
# ----------------------------------------------------------------------------------------------------------------------
class ErroUsuario(Exception):
    """Erro com mensagem que diz ao usuário o que fazer (o programa imprime e sai com código 2)."""


# ----------------------------------------------------------------------------------------------------------------------
# 0.2 ollama_ligado — testa se o servidor do Ollama está respondendo
#     Usada por: 0.3 ligar_ollama
# ----------------------------------------------------------------------------------------------------------------------
def ollama_ligado() -> bool:
    """Diz se o servidor do Ollama está respondendo em ``OLLAMA_HOST``.

    Saídas:
        True se respondeu em até 2 segundos, senão False.
    """
    # Sem proxy: a pergunta vai direto para localhost.
    abridor = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with abridor.open(f"{OLLAMA_HOST}/api/version", timeout=2):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------------------------------------------------------------
# 0.3 ligar_ollama — liga o Ollama se estiver desligado (e desliga ao terminar o programa)
#     Usada por: 0.4 criar_cliente, 7.5 gguf_do_ollama
# ----------------------------------------------------------------------------------------------------------------------
def ligar_ollama() -> None:
    """Liga o servidor do Ollama, se ele ainda não estiver ligado.

    Quando é este programa que liga, o servidor é desligado sozinho ao final (atexit). Se o Ollama já estava ligado
    (por exemplo, pelo aplicativo), nada muda.

    Erros:
        ErroUsuario se o Ollama não estiver instalado ou não ligar em 60 segundos.
    """
    if ollama_ligado():
        return
    if shutil.which("ollama") is None:
        raise ErroUsuario("Ollama não instalado; veja a instalação no README (brew install ollama)")
    print("Ligando o Ollama ...", file=sys.stderr)
    # Roda "ollama serve" em segundo plano, sem misturar a saída dele com a do programa.
    servidor = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(servidor.terminate)
    # Espera o servidor começar a responder.
    for _ in range(60):
        if ollama_ligado():
            return
        time.sleep(1)
    raise ErroUsuario("o Ollama não ligou em 60 segundos; tente rodar `ollama serve` em outro terminal")


# ----------------------------------------------------------------------------------------------------------------------
# 0.4 criar_cliente — cria o cliente do Ollama que só fala com localhost
#     Usada por: 3.4 indexar, 8.2 comando_perguntar, 8.3 comando_responder, avaliador.py
# ----------------------------------------------------------------------------------------------------------------------
def criar_cliente():
    """Liga o Ollama (se preciso) e cria o cliente local.

    Entradas:
        nenhuma (usa ``OLLAMA_HOST``).
    Saídas:
        ``ollama.Client`` apontando para localhost, sem proxy (``trust_env=False``), para garantir que nenhum texto do
        manual seja enviado a outro endereço.
    Erros:
        ErroUsuario se o pacote ``ollama`` não estiver instalado ou o Ollama não ligar.
    """
    # Importa aqui para o --help funcionar mesmo sem o pacote instalado.
    try:
        import ollama
    except ImportError:
        raise ErroUsuario("pacote 'ollama' não instalado; rode: pip install -r requirements.txt") from None
    ligar_ollama()
    # Cliente local, com tempo limite generoso (modelos em CPU são lentos).
    return ollama.Client(host=OLLAMA_HOST, timeout=300, trust_env=False)


# ----------------------------------------------------------------------------------------------------------------------
# 0.5 _chamar_ollama — faz uma chamada ao Ollama; se o modelo faltar, baixa e tenta de novo
#     Usada por: 3.2 gerar_embeddings, 5.2 conversar
# ----------------------------------------------------------------------------------------------------------------------
def _chamar_ollama(funcao, modelo: str):
    """Executa uma chamada ao Ollama; se o modelo não estiver baixado, baixa e tenta de novo.

    Entradas:
        funcao: função sem argumentos que faz a chamada.
        modelo: nome do modelo usado.
    Saídas:
        o que a chamada devolver.
    Erros:
        ErroUsuario se o Ollama estiver desligado ou o modelo não existir no catálogo do Ollama.
    """
    import ollama

    try:
        return funcao()
    except ConnectionError:
        # O pacote ollama converte falha de conexão em ConnectionError.
        raise ErroUsuario(f"Ollama não respondeu em {OLLAMA_HOST}; abra o Ollama ou rode `ollama serve`") from None
    except ollama.ResponseError as erro:
        # 404 = modelo não baixado. Baixa uma vez só (é download do modelo; nenhum texto do manual é enviado).
        if erro.status_code != 404:
            raise
    print(f"Baixando o modelo {modelo} (só na primeira vez; pode demorar) ...", file=sys.stderr)
    try:
        ollama.Client(host=OLLAMA_HOST, trust_env=False).pull(modelo)
    except ollama.ResponseError as erro:
        # "not found" = nome errado; outro erro = falha no download (internet, disco cheio...).
        if "not found" in str(erro).lower():
            raise ErroUsuario(f"modelo {modelo} não existe no Ollama; confira o nome") from None
        raise ErroUsuario(f"não foi possível baixar o modelo {modelo}: {erro}") from None
    except ConnectionError:
        raise ErroUsuario(f"não foi possível baixar o modelo {modelo}; confira a internet") from None
    return funcao()


# ----------------------------------------------------------------------------------------------------------------------
# 0.6 Progresso — barra de progresso no terminal, com tempo decorrido e estimativa do que falta
#     Usada por: 3.2 gerar_embeddings, 6.3 gerar_respostas_propostas, avaliador.py
# ----------------------------------------------------------------------------------------------------------------------
class Progresso:
    """Barra de progresso de uma tarefa longa, redesenhada na mesma linha do terminal (stderr).

    Exemplo de linha: ``Avaliação [████████░░░░░░░░░░░░] 40/200  20% · 8m02s · faltam ~32m · S3 qwen3-manual p. 7``.
    Também funciona com a saída redirecionada para um arquivo: ``tail -f arquivo.log`` mostra a barra andando.

    Uso:
        barra = Progresso(total, "Avaliação")
        for item in itens:
            barra.atualizar("o que está sendo feito agora")   # antes do item
            ...
            barra.avancar()                                   # depois do item
    """

    def __init__(self, total: int, titulo: str):
        """Cria a barra e já a desenha vazia.

        Entradas:
            total: quantos passos a tarefa tem.
            titulo: nome da tarefa, mostrado à esquerda da barra.
        """
        self.total = max(total, 1)
        self.titulo = titulo
        self.feitos = 0
        self.detalhe = ""
        self.inicio = time.perf_counter()
        self._desenhar()

    def atualizar(self, detalhe: str) -> None:
        """Troca o texto do passo em andamento, sem avançar a barra."""
        self.detalhe = detalhe
        self._desenhar()

    def avancar(self, passos: int = 1) -> None:
        """Marca passos como concluídos; ao chegar no total, pula para a linha de baixo."""
        self.feitos = min(self.feitos + passos, self.total)
        if self.feitos == self.total:
            self.detalhe = "concluído"
        self._desenhar()
        if self.feitos == self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    @staticmethod
    def _tempo(segundos: float) -> str:
        """Formata segundos como 45s, 8m02s ou 1h05m."""
        segundos = int(segundos)
        if segundos < 60:
            return f"{segundos}s"
        if segundos < 3600:
            return f"{segundos // 60}m{segundos % 60:02d}s"
        return f"{segundos // 3600}h{segundos % 3600 // 60:02d}m"

    def _desenhar(self) -> None:
        """Escreve a linha da barra por cima da anterior (\\r volta ao início da linha; \\033[K apaga o resto)."""
        fracao = self.feitos / self.total
        cheios = round(LARGURA_BARRA * fracao)
        barra = "█" * cheios + "░" * (LARGURA_BARRA - cheios)
        decorrido = time.perf_counter() - self.inicio
        # Estimativa do que falta: tempo médio por passo até agora x passos restantes.
        falta = f"faltam ~{self._tempo(decorrido / self.feitos * (self.total - self.feitos))}" if self.feitos else "..."
        linha = (
            f"{self.titulo} [{barra}] {self.feitos}/{self.total} {fracao:4.0%} · {self._tempo(decorrido)} · {falta}"
            f" · {self.detalhe}"
        )
        # Corta no tamanho do terminal, para a linha não quebrar (e a barra não se repetir em várias linhas).
        largura = shutil.get_terminal_size((120, 20)).columns
        sys.stderr.write("\r" + linha[: largura - 1] + "\033[K")
        sys.stderr.flush()


# ======================================================================================================================
# ETAPA 1 — LEITURA DO PDF (TEXTO E TABELAS)
#   Entrada: manual.pdf
#   Saída:   lista de elementos em ordem de leitura: blocos de texto e tabelas (linha a linha, célula a célula), cada
#            um com a sua página, sem cabeçalhos e rodapés repetidos
#   Funções: 1.1 limpar · 1.2 ler_tabela · 1.3 ler_elementos
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 1.1 limpar — desfaz a hifenização de fim de linha, troca marcadores de lista por "•" e junta o texto em uma linha
#     Usada por: 1.2 ler_tabela, 1.3 ler_elementos, 2.5 criar_paragrafos
# ----------------------------------------------------------------------------------------------------------------------
def limpar(texto: str) -> str:
    """Limpa o texto de um bloco ou de uma célula de tabela.

    Entradas:
        texto: texto bruto.
    Saídas:
        texto em uma linha: hifenização de fim de linha desfeita ("libera-\\nção" -> "liberação"), marcadores de
        lista de fontes de símbolos (caracteres de uso privado, como U+E006) trocados por "•", quebras de linha
        trocadas por espaço e espaços repetidos colapsados.
    """
    # Normaliza acentos para a forma composta (á em vez de a + ´).
    texto = unicodedata.normalize("NFC", texto)
    # Marcadores de lista vêm de fontes de símbolos, como caracteres de uso privado: vira "•".
    texto = re.sub(r"[-]", "•", texto)
    # Junta palavra quebrada com hífen no fim da linha quando a linha seguinte começa em minúscula.
    texto = re.sub(r"(\w)-\s*\n\s*(?=[a-zà-ÿ])", r"\1", texto)
    # Troca quebras de linha e espaços repetidos por um único espaço.
    return re.sub(r"\s+", " ", texto).strip()


# ----------------------------------------------------------------------------------------------------------------------
# 1.2 ler_tabela — transforma uma tabela achada pelo PyMuPDF em linhas de células já limpas
#     Usada por: 1.3 ler_elementos
# ----------------------------------------------------------------------------------------------------------------------
def ler_tabela(tabela, pagina: int) -> dict | None:
    """Extrai as células de uma tabela, linha por linha.

    Entradas:
        tabela: objeto ``Table`` do PyMuPDF (``page.find_tables()``).
        pagina: número da página.
    Saídas:
        elemento ``{"tipo": "tabela", "pagina", "y", "linhas": [(página, [células])]}``, sem as linhas vazias, ou None
        se a tabela não tiver texto. A página fica em cada linha porque tabelas que continuam na página seguinte são
        juntadas depois (2.3), e cada linha precisa manter a página de onde veio.
    """
    linhas = []
    for celulas in tabela.extract():
        celulas = [limpar(c or "") for c in celulas]
        if any(celulas):
            linhas.append((pagina, celulas))
    if not linhas:
        return None
    return {"tipo": "tabela", "pagina": pagina, "y": tabela.bbox[1], "linhas": linhas}


# ----------------------------------------------------------------------------------------------------------------------
# 1.3 ler_elementos — abre o PDF e devolve textos e tabelas de cada página, em ordem de leitura
#     Usada por: 2.5 criar_paragrafos
# ----------------------------------------------------------------------------------------------------------------------
def ler_elementos(caminho_pdf: Path) -> list[dict]:
    """Lê o PDF página por página, separando tabelas do texto corrido (etapa 1).

    As tabelas são lidas com ``find_tables`` do PyMuPDF, que mantém cada linha da tabela inteira (infração, RIP e
    prazo juntos). O texto que está dentro de uma tabela não é lido de novo como texto corrido: lido bloco a bloco,
    ele sairia embaralhado, com as colunas misturadas.

    Entradas:
        caminho_pdf: caminho do manual em PDF.
    Saídas:
        lista de elementos na ordem de leitura: ``{"tipo": "texto", "pagina", "y", "texto"}`` (texto bruto do bloco)
        ou ``{"tipo": "tabela", ...}`` (ver 1.2).
    Erros:
        ErroUsuario se o PDF não abrir ou não tiver texto selecionável (PDF escaneado).
    """
    try:
        import pymupdf
    except ImportError:
        raise ErroUsuario("pacote 'pymupdf' não instalado; rode: pip install pymupdf") from None

    elementos = []
    try:
        documento = pymupdf.open(caminho_pdf)
    except Exception:
        raise ErroUsuario(f"não foi possível abrir o PDF {caminho_pdf}") from None
    with documento:
        n_paginas = documento.page_count
        for numero, pagina in enumerate(documento, start=1):
            # Tabelas da página (pelas linhas de grade desenhadas no PDF).
            tabelas = pagina.find_tables().tables
            areas = [pymupdf.Rect(t.bbox) for t in tabelas]
            for tabela in tabelas:
                elemento = ler_tabela(tabela, numero)
                if elemento:
                    elementos.append(elemento)
            # Blocos de texto: cada bloco é (x0, y0, x1, y1, texto, nº do bloco, tipo); tipo 0 = texto, 1 = imagem.
            for x0, y0, x1, y1, texto, _, tipo in pagina.get_text("blocks"):
                if tipo != 0 or not texto.strip():
                    continue
                # Pula o bloco se o centro dele cair dentro de uma tabela (já lido célula a célula).
                centro = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
                if any(centro in area for area in areas):
                    continue
                elementos.append({"tipo": "texto", "pagina": numero, "y": y0, "x": x0, "texto": texto})
    # Ordena por página e de cima para baixo (e da esquerda para a direita, no caso de empate).
    elementos.sort(key=lambda e: (e["pagina"], round(e["y"]), e.get("x", 0)))

    # PDF escaneado não tem camada de texto: menos de 50 caracteres por página em média.
    total = sum(len(e["texto"]) for e in elementos if e["tipo"] == "texto")
    total += sum(len(c) for e in elementos if e["tipo"] == "tabela" for _, linha in e["linhas"] for c in linha)
    if n_paginas == 0 or total < 50 * n_paginas:
        raise ErroUsuario("PDF sem texto selecionável; rode OCR antes (ex.: ocrmypdf manual.pdf manual_ocr.pdf)")

    # Cabeçalho/rodapé: o mesmo texto (ignorando números) em mais da metade das páginas.
    def chave(texto: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"\d", "", texto)).strip().lower()

    paginas_por_chave: dict[str, set[int]] = {}
    for e in elementos:
        if e["tipo"] == "texto":
            paginas_por_chave.setdefault(chave(e["texto"]), set()).add(e["pagina"])
    repetidos = {c for c, pags in paginas_por_chave.items() if len(pags) >= 2 and len(pags) > 0.5 * n_paginas}
    # Devolve tudo, menos os blocos de texto que são cabeçalho ou rodapé.
    return [e for e in elementos if e["tipo"] == "tabela" or chave(e["texto"]) not in repetidos]


# ======================================================================================================================
# ETAPA 2 — DIVISÃO EM PARÁGRAFOS E TRECHOS
#   Entrada: elementos da etapa 1
#   Saída:   trechos de até ~400 tokens, cada um com seção e páginas. O texto corrido vira parágrafos inteiros; cada
#            linha de tabela vira um parágrafo próprio no formato "coluna: valor", com o nome da tabela na frente
#   Funções: 2.1 contar_tokens · 2.2 numero_plausivel · 2.3 juntar_tabelas_continuadas · 2.4 paragrafos_da_tabela
#            2.5 criar_paragrafos · 2.6 dividir_paragrafo · 2.7 criar_trechos
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 2.1 contar_tokens — estima quantos tokens um texto ocupa (palavras x 2,0)
#     Usada por: 2.6 dividir_paragrafo, 2.7 criar_trechos
# ----------------------------------------------------------------------------------------------------------------------
def contar_tokens(texto: str) -> int:
    """Estima o tamanho de um texto em tokens do modelo leitor.

    Entradas:
        texto: qualquer texto.
    Saídas:
        número de palavras x ``TOKENS_POR_PALAVRA`` (medido com o tokenizador do Qwen3 sobre o próprio manual).
    """
    return round(len(texto.split()) * TOKENS_POR_PALAVRA)


# ----------------------------------------------------------------------------------------------------------------------
# 2.2 numero_plausivel — decide se um número de título continua a numeração das seções
#     Usada por: 2.5 criar_paragrafos
# ----------------------------------------------------------------------------------------------------------------------
def numero_plausivel(novo: str, atual: str) -> bool:
    """Confere se a numeração de um título pode vir depois da seção atual.

    Evita que passos numerados ("1 Acione o alarme") ou linhas de tabela ("202 Armazém 3") virem seções.

    Entradas:
        novo: numeração do candidato a título (ex.: "2.5").
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


# ----------------------------------------------------------------------------------------------------------------------
# 2.3 juntar_tabelas_continuadas — junta a parte de uma tabela que continua na página seguinte
#     Usada por: 2.5 criar_paragrafos
# ----------------------------------------------------------------------------------------------------------------------
def juntar_tabelas_continuadas(elementos: list[dict]) -> list[dict]:
    """Junta tabelas quebradas entre páginas.

    Uma tabela continua na página seguinte quando é o primeiro elemento dessa página (logo depois de outra tabela, sem
    texto no meio) e tem o mesmo número de colunas. Se a primeira linha da continuação começa em minúscula ou com a
    primeira célula vazia, ela é o resto da última linha da página anterior (uma célula quebrada na virada da página)
    e é colada nela, célula por célula.

    Entradas:
        elementos: saída de ``ler_elementos`` (1.3).
    Saídas:
        a mesma lista, com as continuações incorporadas à tabela de origem.
    """
    saida: list[dict] = []
    for e in elementos:
        anterior = saida[-1] if saida else None
        continua = (
            e["tipo"] == "tabela"
            and anterior is not None
            and anterior["tipo"] == "tabela"
            and e["pagina"] == anterior["linhas"][-1][0] + 1
            and len(e["linhas"][0][1]) == len(anterior["linhas"][-1][1])
        )
        if not continua:
            saida.append(e)
            continue
        linhas = list(e["linhas"])
        primeira = linhas[0][1]
        texto_inicial = next((c for c in primeira if c), "")
        if not primeira[0] or texto_inicial[:1].islower():
            # Cola cada célula da primeira linha no fim da célula correspondente da última linha anterior.
            pagina_ultima, ultima = anterior["linhas"][-1]
            colada = [f"{a} {b}".strip() for a, b in zip(ultima, primeira, strict=True)]
            anterior["linhas"][-1] = (pagina_ultima, colada)
            linhas = linhas[1:]
        anterior["linhas"].extend(linhas)
    return saida


# ----------------------------------------------------------------------------------------------------------------------
# 2.4 paragrafos_da_tabela — transforma cada linha de uma tabela em um parágrafo "coluna: valor; coluna: valor"
#     Usada por: 2.5 criar_paragrafos
# ----------------------------------------------------------------------------------------------------------------------
def paragrafos_da_tabela(tabela: dict, legenda: str | None) -> list[tuple[int, str]]:
    """Escreve as linhas de uma tabela como parágrafos que se entendem sozinhos.

    Cada linha leva o nome das colunas (do cabeçalho) e, na frente, a tabela e o grupo a que pertence. Assim a busca
    e o leitor encontram, por exemplo, "[Tabela 3, INFRAÇÕES DO ARRENDATÁRIO] Não conformidade: ...; RIP: NÃO; Prazo:
    0 dia: ..." mesmo quando o trecho não contém o cabeçalho da tabela.

    Entradas:
        tabela: elemento de tabela (1.2), já com as continuações juntadas (2.3).
        legenda: legenda da tabela no texto ("Tabela 3 – ..."), se houver.
    Saídas:
        lista de (página, texto do parágrafo).
    """
    # Da legenda fica só "Tabela N": a legenda completa já entra no texto corrido, logo antes da tabela.
    nome = re.match(r"Tabela\s+\d+", legenda).group(0) if legenda and re.match(r"Tabela\s+\d+", legenda) else None
    cabecalho: list[str] | None = None
    grupo: str | None = None
    vistas_dados = False
    saida = []
    for pagina, celulas in tabela["linhas"]:
        preenchidas = [c for c in celulas if c]
        # Linha com uma célula só e curta (numa tabela de várias colunas) é título de grupo: "INFRAÇÕES DO
        # ARRENDATÁRIO". Frases longas sozinhas numa linha (comuns nos formulários) são conteúdo.
        if len(celulas) > 1 and len(preenchidas) == 1 and len(preenchidas[0]) <= 80:
            grupo = preenchidas[0]
            continue
        # Cabeçalho: antes da primeira linha de dados, todas as células preenchidas e curtas ("Não conformidade").
        if not vistas_dados and all(celulas) and all(len(c) <= 40 for c in celulas):
            cabecalho = celulas
            continue
        vistas_dados = True
        if cabecalho and len(cabecalho) == len(celulas):
            texto = "; ".join(f"{titulo}: {c}" for titulo, c in zip(cabecalho, celulas, strict=True) if c)
        else:
            texto = " | ".join(preenchidas)
        contexto = ", ".join(x for x in (nome, grupo) if x)
        saida.append((pagina, f"[{contexto}] {texto}" if contexto else texto))
    return saida


# ----------------------------------------------------------------------------------------------------------------------
# 2.5 criar_paragrafos — percorre os elementos acompanhando a seção corrente e monta os parágrafos
#     Usada por: 2.7 criar_trechos
# ----------------------------------------------------------------------------------------------------------------------
def criar_paragrafos(caminho_pdf: Path) -> list[dict]:
    """Transforma o PDF em parágrafos com seção e páginas (etapas 1 e 2).

    Neste tipo de PDF cada linha costuma ser um bloco separado; por isso as linhas são juntadas até o fim da frase
    (ponto final), em vez de cada bloco virar um parágrafo. Títulos numerados ("1.4. BASE LEGAL E NORMATIVA", com o
    número na linha de cima) abrem uma seção nova e também entram no texto, para a busca achar a seção pelo nome.

    Entradas:
        caminho_pdf: caminho do manual.
    Saídas:
        lista de ``{"secao", "secao_id", "pagina", "pagina_fim", "texto"}``.
    Erros:
        ErroUsuario se o PDF não tiver texto ou não gerar nenhum parágrafo.
    """
    # Etapa 1: elementos do PDF; tabelas quebradas entre páginas voltam a ser uma tabela só.
    elementos = juntar_tabelas_continuadas(ler_elementos(caminho_pdf))

    paragrafos: list[dict] = []
    secao, secao_id = "Sem título", ""
    legenda = None
    atual: dict | None = None

    def novo(pagina: int, texto: str) -> dict:
        return {"secao": secao, "secao_id": secao_id, "pagina": pagina, "pagina_fim": pagina, "texto": texto}

    def fechar() -> None:
        nonlocal atual
        if atual:
            paragrafos.append(atual)
        atual = None

    for e in elementos:
        # Tabela: cada linha vira um parágrafo próprio, na seção corrente.
        if e["tipo"] == "tabela":
            fechar()
            paragrafos.extend(novo(pagina, texto) for pagina, texto in paragrafos_da_tabela(e, legenda))
            continue
        # Descarta linhas de sumário ("4.2 Liberação de carga ........ 17").
        linhas = [linha for linha in e["texto"].splitlines() if linha.strip() and not REGEX_SUMARIO.search(linha)]
        bloco = limpar("\n".join(linhas))
        # Sobra só numeração (número de página, número de item do sumário): não é conteúdo.
        if not bloco or REGEX_SO_NUMERO.match(bloco):
            continue
        # Título numerado em sequência: abre uma nova seção e vira um parágrafo curto com o nome dela.
        titulo = REGEX_TITULO.match(bloco)
        # Número de um nível só ("5. O operador deverá...") só é título se estiver sozinho na linha de cima (como nos
        # títulos deste manual) ou se o título estiver em maiúsculas ("4 ANEXOS"): assim itens numerados de listas e
        # anexos não viram seções.
        numero_sozinho = REGEX_SO_NUMERO.match(linhas[0])
        if titulo and "." not in titulo.group(1) and not numero_sozinho and not titulo.group(2).isupper():
            titulo = None
        if titulo and numero_plausivel(titulo.group(1), secao_id):
            fechar()
            secao_id, secao = titulo.group(1), f"{titulo.group(1)} {titulo.group(2)}"
            # A legenda vale para as tabelas da seção em que aparece (a Tabela 3 volta a aparecer depois de uma
            # quebra, sem repetir a legenda); uma seção nova a esquece.
            legenda = None
            paragrafos.append(novo(e["pagina"], secao))
            continue
        # Legenda de tabela ("Tabela 3 – ..."): guarda para nomear as linhas da próxima tabela.
        if REGEX_LEGENDA.match(bloco):
            legenda = bloco
        # Texto corrido: continua o parágrafo aberto (desfazendo hífen de fim de linha) ou abre um novo.
        if atual is None:
            atual = novo(e["pagina"], bloco)
        elif atual["texto"].endswith("-") and bloco[:1].islower():
            atual["texto"] = atual["texto"][:-1] + bloco
        else:
            atual["texto"] += " " + bloco
        atual["pagina_fim"] = e["pagina"]
        # Fim de frase (ou legenda de tabela) fecha o parágrafo.
        if bloco.endswith((".", "!", "?")) or REGEX_LEGENDA.match(bloco):
            fechar()
    fechar()
    if not paragrafos:
        raise ErroUsuario(f"nenhum parágrafo extraído de {caminho_pdf}; confira se o PDF tem texto")
    return paragrafos


# ----------------------------------------------------------------------------------------------------------------------
# 2.6 dividir_paragrafo — corta um parágrafo maior que o limite do trecho em pedaços, entre frases
#     Usada por: 2.7 criar_trechos
# ----------------------------------------------------------------------------------------------------------------------
def dividir_paragrafo(paragrafo: dict) -> list[dict]:
    """Divide um parágrafo longo demais em pedaços de até ``MAX_TOKENS_TRECHO`` tokens, sem cortar frases.

    Entradas:
        paragrafo: um parágrafo de ``criar_paragrafos`` (2.5).
    Saídas:
        lista de parágrafos (o próprio, se já couber). Uma frase sozinha maior que o limite fica inteira.
    """
    if contar_tokens(paragrafo["texto"]) <= MAX_TOKENS_TRECHO:
        return [paragrafo]
    pedacos, atual = [], ""
    # Corta depois de ponto, ponto e vírgula ou dois-pontos seguidos de espaço.
    for frase in re.split(r"(?<=[.;:])\s+", paragrafo["texto"]):
        if atual and contar_tokens(f"{atual} {frase}") > MAX_TOKENS_TRECHO:
            pedacos.append(atual)
            atual = frase
        else:
            atual = f"{atual} {frase}".strip()
    pedacos.append(atual)
    return [{**paragrafo, "texto": texto} for texto in pedacos]


# ----------------------------------------------------------------------------------------------------------------------
# 2.7 criar_trechos — agrupa parágrafos seguidos em trechos de até ~400 tokens
#     Usada por: 3.4 indexar
# ----------------------------------------------------------------------------------------------------------------------
def criar_trechos(caminho_pdf: Path) -> list[dict]:
    """Transforma o PDF em trechos com seção e página (etapas 1 e 2).

    Entradas:
        caminho_pdf: caminho do manual.
    Saídas:
        lista de trechos ``{"id", "secao", "secao_id", "pagina_inicio", "pagina_fim", "texto"}``. Cada trecho junta
        parágrafos inteiros até ~400 tokens. Uma seção nova começa um trecho novo, a não ser que o trecho atual ainda
        esteja pequeno (menos de ``MIN_TOKENS_TRECHO``): assim subseções de uma linha só não viram trechos soltos.
    Erros:
        ErroUsuario (via 2.5).
    """
    # Etapas 1 e 2.5: parágrafos; 2.6: os longos demais são divididos entre frases.
    paragrafos = [p for paragrafo in criar_paragrafos(caminho_pdf) for p in dividir_paragrafo(paragrafo)]

    grupos: list[list[dict]] = []
    for paragrafo in paragrafos:
        ultimo = grupos[-1] if grupos else None
        tamanho = sum(contar_tokens(p["texto"]) for p in ultimo) if ultimo else 0
        # Começa um grupo novo se o parágrafo não cabe, ou se mudou a seção e o grupo atual já tem tamanho bom.
        cabe = ultimo and tamanho + contar_tokens(paragrafo["texto"]) <= MAX_TOKENS_TRECHO
        mesma_secao = ultimo and ultimo[-1]["secao"] == paragrafo["secao"]
        if cabe and (mesma_secao or tamanho < MIN_TOKENS_TRECHO):
            ultimo.append(paragrafo)
        else:
            grupos.append([paragrafo])

    # Converte cada grupo em um trecho com id sequencial (t0001, t0002...). A seção do trecho é a do primeiro
    # parágrafo de conteúdo (o título de uma seção nova, quando o trecho começa nele).
    return [
        {
            "id": f"t{numero:04d}",
            "secao": grupo[0]["secao"],
            "secao_id": grupo[0]["secao_id"],
            "pagina_inicio": grupo[0]["pagina"],
            "pagina_fim": max(p["pagina_fim"] for p in grupo),
            "texto": "\n".join(p["texto"] for p in grupo),
        }
        for numero, grupo in enumerate(grupos, start=1)
    ]


# ======================================================================================================================
# ETAPA 3 — INDEXAÇÃO (BM25 E EMBEDDINGS)
#   Entrada: trechos da etapa 2
#   Saída:   pasta indice/ (trechos.json, vetores.npy, manual.sha256) e o objeto Indice na memória
#   Funções: 3.1 tokenizar · 3.2 gerar_embeddings · 3.3 impressao_digital · 3.4 indexar · 3.5 Indice
#            3.6 carregar_indice
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 3.1 tokenizar — texto -> palavras para o BM25 (minúsculas, sem stopwords)
#     Usada por: 3.5 Indice, 4.1 buscar_bm25
# ----------------------------------------------------------------------------------------------------------------------
def tokenizar(texto: str) -> list[str]:
    """Divide um texto em palavras para o BM25.

    Entradas:
        texto: qualquer texto (trecho ou pergunta).
    Saídas:
        palavras em minúsculas, sem pontuação, sem palavras de 1 letra e sem stopwords (acentos mantidos).
    """
    palavras = re.findall(r"\w+", unicodedata.normalize("NFC", texto.lower()))
    return [p for p in palavras if len(p) > 1 and p not in STOPWORDS]


# ----------------------------------------------------------------------------------------------------------------------
# 3.2 gerar_embeddings — textos -> vetores do bge-m3, normalizados (busca por significado)
#     Usada por: 3.4 indexar, 4.2 buscar_denso
# ----------------------------------------------------------------------------------------------------------------------
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
    # Barra de progresso só na indexação (muitos textos); a pergunta da busca densa é um texto só.
    barra = Progresso(len(textos), "Embeddings") if len(textos) > 16 else None
    # Envia em lotes de 16 para não estourar a memória do Ollama em CPU.
    for inicio in range(0, len(textos), 16):
        lote = textos[inicio : inicio + 16]
        if barra:
            barra.atualizar(f"trechos {inicio + 1} a {inicio + len(lote)}")
        resposta = _chamar_ollama(
            lambda lote=lote: cliente.embed(model=MODELO_EMBEDDINGS, input=lote), MODELO_EMBEDDINGS
        )
        vetores.extend(resposta.embeddings)
        if barra:
            barra.avancar(len(lote))
    matriz = np.asarray(vetores, dtype=np.float32)
    # Normaliza cada linha (evita divisão por zero em vetores nulos).
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


# ----------------------------------------------------------------------------------------------------------------------
# 3.3 impressao_digital — SHA-256 do manual + versão da divisão, para saber se o índice precisa ser refeito
#     Usada por: 3.4 indexar, 3.6 carregar_indice
# ----------------------------------------------------------------------------------------------------------------------
def impressao_digital(caminho: Path) -> str:
    """Identifica o manual e o jeito de dividi-lo (muda se o manual mudar ou se ``VERSAO_INDICE`` aumentar).

    Entradas:
        caminho: arquivo a ler.
    Saídas:
        o SHA-256 do conteúdo, em hexadecimal, seguido de " v" e a versão da divisão em trechos.
    """
    resumo = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        # Lê em blocos de 1 MB para não carregar o PDF inteiro na memória.
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return f"{resumo.hexdigest()} v{VERSAO_INDICE}"


# ----------------------------------------------------------------------------------------------------------------------
# 3.4 indexar — roda as etapas 1, 2 e 3 e grava o índice em indice/
#     Usada por: 3.6 carregar_indice, 8.1 comando_indexar
# ----------------------------------------------------------------------------------------------------------------------
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
    ARQ_HASH.write_text(impressao_digital(caminho_pdf), encoding="utf-8")
    return len(trechos)


# ----------------------------------------------------------------------------------------------------------------------
# 3.5 Indice — guarda trechos, vetores e o índice BM25 na memória
#     Usada por: 3.6 carregar_indice
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 3.6 carregar_indice — lê indice/ (e indexa antes, se faltar ou se o manual mudou)
#     Usada por: 8.2 comando_perguntar, 8.3 comando_responder, 8.4 comando_treinar, avaliador.py
# ----------------------------------------------------------------------------------------------------------------------
def carregar_indice() -> Indice:
    """Lê o índice gravado em ``indice/``, criando-o antes se ainda não existir ou se o manual.pdf mudou.

    Saídas:
        objeto ``Indice`` pronto para buscas.
    Erros:
        ErroUsuario se não houver índice nem manual.pdf para criá-lo.
    """
    import numpy as np

    existe = ARQ_TRECHOS.exists() and ARQ_VETORES.exists()
    # Conteúdo do manual diferente do indexado = o manual foi trocado; o índice precisa ser refeito.
    desatualizado = (
        existe
        and ARQ_MANUAL.exists()
        and (not ARQ_HASH.exists() or ARQ_HASH.read_text(encoding="utf-8") != impressao_digital(ARQ_MANUAL))
    )
    if not existe or desatualizado:
        if not ARQ_MANUAL.exists():
            raise ErroUsuario(
                f"manual não encontrado; coloque o arquivo {ARQ_MANUAL.name} ao lado de {Path(__file__).name}"
            )
        print("Indexando o manual (só na primeira vez ou quando o manual muda) ...", file=sys.stderr)
        print(f"Índice criado com {indexar(ARQ_MANUAL)} trechos.", file=sys.stderr)
    trechos = json.loads(ARQ_TRECHOS.read_text(encoding="utf-8"))
    return Indice(trechos, np.load(ARQ_VETORES))


# ======================================================================================================================
# ETAPA 4 — BUSCA (BM25, DENSA E HÍBRIDA COM RRF)
#   Entrada: pergunta + índice da etapa 3
#   Saída:   ids dos 5 trechos mais relevantes
#   Funções: 4.1 buscar_bm25 · 4.2 buscar_denso · 4.3 fundir_rrf · 4.4 buscar
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 4.1 buscar_bm25 — busca por palavras em comum (S1)
#     Usada por: 4.4 buscar
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 4.2 buscar_denso — busca por significado: cosseno entre embeddings (S2)
#     Usada por: 4.4 buscar
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 4.3 fundir_rrf — junta as listas do BM25 e da busca densa por Reciprocal Rank Fusion (S3)
#     Usada por: 4.4 buscar
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 4.4 buscar — escolhe o modo de busca: S0 nenhum, S1 BM25, S2 denso, S3 híbrido
#     Usada por: 5.6 responder
# ----------------------------------------------------------------------------------------------------------------------
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
# ETAPA 5 — RESPOSTA E VERIFICAÇÃO DA CITAÇÃO
#   Entrada: pergunta + trechos da etapa 4
#   Saída:   resposta com citação "(seção X, p. N)", conferida contra os trechos
#   Funções: 5.1 montar_prompt · 5.2 conversar · 5.3 extrair_citacoes · 5.4 citacao_valida · 5.5 absteve · 5.6 responder
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 5.1 montar_prompt — monta o texto enviado ao leitor, com os trechos numerados
#     Usada por: 5.6 responder
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 5.2 conversar — envia o prompt ao modelo leitor e devolve o texto da resposta
#     Usada por: 5.6 responder
# ----------------------------------------------------------------------------------------------------------------------
def conversar(cliente, modelo: str, prompt: str) -> str:
    """Pede uma resposta ao modelo de linguagem.

    Entradas:
        cliente: cliente do Ollama.
        modelo: nome do modelo no Ollama.
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


# ----------------------------------------------------------------------------------------------------------------------
# 5.3 extrair_citacoes — acha as citações (seção, página) no texto da resposta
#     Usada por: 5.6 responder
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 5.4 citacao_valida — confere se a citação aponta para um dos trechos usados
#     Usada por: 5.6 responder, 6.3 gerar_respostas_propostas
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 5.5 absteve — diz se a resposta é "Não encontrado no manual"
#     Usada por: 5.6 responder, avaliador.py
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 5.6 responder — pergunta -> busca (etapa 4) -> resposta e citação (etapa 5)
#     Usada por: 8.2 comando_perguntar, 6.3 gerar_respostas_propostas, avaliador.py
# ----------------------------------------------------------------------------------------------------------------------
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
# ETAPA 6 — RESPOSTAS PROPOSTAS PARA O GABARITO (comando responder)
#   Entrada: perguntas.json
#   Saída:   resposta.json, com uma resposta proposta por pergunta para revisão humana
#   Funções: 6.1 ler_perguntas · 6.2 tipo_da_pergunta · 6.3 gerar_respostas_propostas
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 6.1 ler_perguntas — lê e valida o perguntas.json
#     Usada por: 8.3 comando_responder
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 6.2 tipo_da_pergunta — classifica a pergunta: factual, procedimental ou sem_resposta
#     Usada por: 6.3 gerar_respostas_propostas
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 6.3 gerar_respostas_propostas — responde cada pergunta (5.6) e monta os itens do resposta.json
#     Usada por: 8.3 comando_responder
# ----------------------------------------------------------------------------------------------------------------------
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
    barra = Progresso(len(perguntas), "Respostas")
    for item in perguntas:
        barra.atualizar(f"pergunta {item['id']}")
        resultado = responder(item["pergunta"], indice, cliente, "hibrido", modelo)
        barra.avancar()
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
# ETAPA 7 — AJUSTE FINO LoRA (comando treinar)
#   Entrada: índice da etapa 3
#   Saída:   modelo qwen3-manual registrado no Ollama
#   Funções: 7.1 rodar · 7.2 preparar_dados · 7.3 treinar · 7.4 converter_adaptador · 7.5 gguf_do_ollama
#            7.6 juntar_no_leitor · 7.7 registrar_no_ollama
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 7.1 rodar — roda um programa externo e para com erro se ele falhar
#     Usada por: 7.3 treinar, 7.4 converter_adaptador, 7.6 juntar_no_leitor, 7.7 registrar_no_ollama
# ----------------------------------------------------------------------------------------------------------------------
def rodar(comando: list[str], etapa: str) -> None:
    """Roda um comando externo mostrando a saída dele no terminal.

    Entradas:
        comando: programa e argumentos.
        etapa: nome da etapa, para a mensagem de erro.
    Erros:
        ErroUsuario se o comando terminar com erro.
    """
    if subprocess.run(comando, check=False).returncode != 0:
        raise ErroUsuario(f"a etapa '{etapa}' falhou (veja a mensagem acima)")


# ----------------------------------------------------------------------------------------------------------------------
# 7.2 preparar_dados — passo 1 do treino: trechos -> conversas em modelos/dados/train.jsonl
#     Usada por: 8.4 comando_treinar
# ----------------------------------------------------------------------------------------------------------------------
def preparar_dados(indice: Indice) -> int:
    """Passo 1 do treino: grava o manual inteiro, trecho por trecho, como conversas no formato do mlx-lm.

    Cada trecho vira uma conversa curta: o usuário pede o conteúdo da seção e o modelo responde com o trecho. O
    formato de conversa (em vez de texto corrido) mantém o modelo respondendo e encerrando a resposta como antes;
    treinado com texto corrido, ele passa a "continuar o documento" e entra em repetição.

    Entradas:
        indice: índice do manual.
    Saídas:
        número de trechos gravados em ``modelos/dados/train.jsonl`` (uma conversa por linha). Não há conjunto de
        validação: o treino usa o manual todo.
    """
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    # Apaga uma validação antiga, se houver, para o mlx-lm não usá-la.
    (PASTA_DADOS / "valid.jsonl").unlink(missing_ok=True)
    with open(PASTA_DADOS / "train.jsonl", "w", encoding="utf-8") as arquivo:
        for trecho in indice.trechos:
            # A pergunta cita seção e página, para o modelo associar cada assunto ao lugar dele no manual.
            pedido = f"O que diz o manual na seção {trecho['secao']} (p. {trecho['pagina_inicio']})?"
            conversa = [{"role": "user", "content": pedido}, {"role": "assistant", "content": trecho["texto"]}]
            arquivo.write(json.dumps({"messages": conversa}, ensure_ascii=False) + "\n")
    return len(indice.trechos)


# ----------------------------------------------------------------------------------------------------------------------
# 7.3 treinar — passo 2 do treino: treina o adaptador LoRA com o mlx-lm
#     Usada por: 8.4 comando_treinar
# ----------------------------------------------------------------------------------------------------------------------
def treinar(n_treino: int, epocas: float) -> None:
    """Passo 2 do treino: treina o adaptador LoRA com o mlx-lm.

    Entradas:
        n_treino: número de exemplos de treino.
        epocas: quantas vezes cada exemplo é visto.
    Saídas:
        nenhuma; grava o adaptador em ``modelos/adaptador/``.
    """
    # Um exemplo por passo (lote 1, para caber na memória), acumulando 4 passos por atualização dos pesos.
    iteracoes = max(1, math.ceil(n_treino * epocas))
    rodar(
        [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", MODELO_BASE_MLX,
            "--train",
            "--mask-prompt",  # aprende só a resposta (o trecho), não o pedido
            "--data", str(PASTA_DADOS),
            "--adapter-path", str(PASTA_ADAPTADOR),
            "--iters", str(iteracoes),
            "--batch-size", "1",
            "--grad-accumulation-steps", "4",
            "--num-layers", str(CAMADAS_LORA),
            "--learning-rate", str(TAXA_APRENDIZADO),
            "--max-seq-length", str(MAX_TOKENS_TREINO),
            "--grad-checkpoint",
            "--seed", str(SEMENTE_TREINO),
        ],
        "treino",
    )  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------------
# 7.4 converter_adaptador — passo 3 do treino: adaptador MLX -> PEFT -> GGUF
#     Usada por: 8.4 comando_treinar
# ----------------------------------------------------------------------------------------------------------------------
def converter_adaptador() -> None:
    """Passo 3 do treino: converte o adaptador do formato do MLX para GGUF.

    O conversor do llama.cpp lê adaptadores no formato PEFT (Hugging Face). Os dois formatos guardam as mesmas
    matrizes A e B do LoRA, só com outros nomes e transpostas; aqui elas são renomeadas e transpostas.

    Saídas:
        nenhuma; grava ``modelos/adaptador.gguf`` e apaga do cache os pesos MLX baixados para o treino (~2,3 GB).
    """
    import numpy as np
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from safetensors.numpy import load_file, save_file

    # Posto e escala do LoRA usados no treino (no PEFT a escala vira lora_alpha = escala x posto).
    parametros = json.loads((PASTA_ADAPTADOR / "adapter_config.json").read_text(encoding="utf-8"))["lora_parameters"]
    # MLX: "model.layers.N.mlp.down_proj.lora_a" (entrada x posto).
    # PEFT: "base_model.model.model.layers.N.mlp.down_proj.lora_A.weight" (posto x entrada).
    pesos = {}
    for nome, matriz in load_file(PASTA_ADAPTADOR / "adapters.safetensors").items():
        camada, tipo = nome.rsplit(".", 1)
        pesos[f"base_model.model.{camada}.lora_{tipo[-1].upper()}.weight"] = np.ascontiguousarray(matriz.T)
    PASTA_PEFT.mkdir(parents=True, exist_ok=True)
    save_file(pesos, PASTA_PEFT / "adapter_model.safetensors")
    # O conversor precisa só da configuração do modelo base (arquivos .json, já no cache desde o treino).
    base = snapshot_download(MODELO_BASE_MLX, allow_patterns=["*.json"])
    configuracao = {
        "peft_type": "LORA",
        "base_model_name_or_path": base,
        "r": parametros["rank"],
        "lora_alpha": parametros["scale"] * parametros["rank"],
        "target_modules": sorted({nome.split(".")[-3] for nome in pesos}),
    }
    (PASTA_PEFT / "adapter_config.json").write_text(json.dumps(configuracao, indent=1), encoding="utf-8")
    comando = [sys.executable, str(CONVERSOR_LORA), str(PASTA_PEFT), "--base", base]
    rodar([*comando, "--outfile", str(ARQ_ADAPTADOR_GGUF), "--outtype", "f16"], "conversão do adaptador")
    # Os pesos MLX só serviam para o treino: apaga do cache do Hugging Face para liberar espaço.
    shutil.rmtree(Path(HF_HUB_CACHE) / ("models--" + MODELO_BASE_MLX.replace("/", "--")), ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------------
# 7.5 gguf_do_ollama — acha o arquivo GGUF que o Ollama usa para um modelo
#     Usada por: 7.6 juntar_no_leitor, avaliador.py
# ----------------------------------------------------------------------------------------------------------------------
def gguf_do_ollama(modelo: str) -> Path:
    """Acha o arquivo GGUF que o Ollama usa para um modelo (base do ajuste fino e da perplexidade).

    Entradas:
        modelo: nome no Ollama.
    Saídas:
        caminho do arquivo (o "blob" do Ollama é um GGUF válido para o llama.cpp).
    Erros:
        ErroUsuario se o comando ollama não existir ou o modelo não estiver instalado.
    """
    if shutil.which("ollama") is None:
        raise ErroUsuario("comando ollama não encontrado; informe o arquivo com --gguf modelo=arquivo.gguf")
    # O "ollama show" precisa do servidor ligado.
    ligar_ollama()
    # O Modelfile mostrado pelo Ollama tem uma linha "FROM /caminho/do/blob".
    comando = ["ollama", "show", modelo, "--modelfile"]
    saida = subprocess.run(comando, capture_output=True, text=True, check=False)
    if saida.returncode != 0:
        # Modelo ainda não baixado: baixa e tenta de novo.
        subprocess.run(["ollama", "pull", modelo], check=False)
        saida = subprocess.run(comando, capture_output=True, text=True, check=False)
    if saida.returncode != 0:
        raise ErroUsuario(f"modelo {modelo} não encontrado no Ollama")
    for caminho in re.findall(r"^FROM\s+(\S.*)$", saida.stdout, re.M):
        if Path(caminho.strip()).is_file():
            return Path(caminho.strip())
    raise ErroUsuario(f"não encontrei o arquivo GGUF de {modelo} na saída de `ollama show`")


# ----------------------------------------------------------------------------------------------------------------------
# 7.6 juntar_no_leitor — passo 4 do treino: soma o adaptador ao leitor e compacta em 4 bits
#     Usada por: 8.4 comando_treinar
# ----------------------------------------------------------------------------------------------------------------------
def juntar_no_leitor() -> None:
    """Passo 4 do treino: soma o adaptador ao GGUF do leitor que o Ollama já tem e compacta o resultado.

    Saídas:
        nenhuma; grava ``modelos/qwen3-manual.gguf``, com cada matriz no mesmo tipo (4, 6 ou 16 bits) que ela tem no
        leitor original. Assim o modelo ajustado tem o mesmo tamanho do original e a comparação entre os dois é justa.
    """
    # Usa o próprio arquivo do Ollama como base: nada de baixar ou converter o modelo inteiro de novo.
    base = gguf_do_ollama(MODELO_LEITOR)
    # Soma o adaptador: as matrizes treinadas saem em 16 bits; as outras são copiadas sem mudança.
    rodar([str(EXPORT_LORA), "-m", str(base), "--lora", str(ARQ_ADAPTADOR_GGUF), "-o", str(ARQ_JUNTO)], "junção")
    # Lista o tipo original de cada matriz (lendo só o cabeçalho do GGUF do Ollama, com a biblioteca do llama.cpp).
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    from gguf import GGUFReader

    tipos = [f"^{re.escape(t.name)}$={t.tensor_type.name}" for t in GGUFReader(str(base)).tensors]
    ARQ_TIPOS.write_text("\n".join(tipos) + "\n", encoding="utf-8")
    # Devolve cada matriz ao tipo original. O llama-quantize só mexe nas que mudaram de tipo (as treinadas).
    comando = [str(QUANTIZE), "--allow-requantize", "--tensor-type-file", str(ARQ_TIPOS)]
    rodar([*comando, str(ARQ_JUNTO), str(ARQ_GGUF), "Q4_K_M"], "compactação")
    ARQ_JUNTO.unlink()


# ----------------------------------------------------------------------------------------------------------------------
# 7.7 registrar_no_ollama — passo 5 do treino: registra o qwen3-manual no Ollama
#     Usada por: 8.4 comando_treinar
# ----------------------------------------------------------------------------------------------------------------------
def registrar_no_ollama() -> None:
    """Passo 5 do treino: cria o modelo ``qwen3-manual`` no Ollama a partir do GGUF junto.

    Copia o Modelfile do leitor (formato de conversa e parâmetros) trocando só o arquivo de pesos, para que o modelo
    ajustado converse do mesmo jeito que o original.

    Saídas:
        nenhuma; o modelo fica disponível no Ollama e o GGUF local é apagado (o Ollama guarda uma cópia).
    """
    comando = ["ollama", "show", MODELO_LEITOR, "--modelfile"]
    saida = subprocess.run(comando, capture_output=True, text=True, check=False)
    if saida.returncode != 0:
        raise ErroUsuario(f"não consegui ler o Modelfile de {MODELO_LEITOR} no Ollama")
    # Troca a linha "FROM <pesos do leitor>" pelo GGUF ajustado.
    modelfile = re.sub(r"^FROM .*$", f"FROM {ARQ_GGUF}", saida.stdout, count=1, flags=re.MULTILINE)
    ARQ_MODELFILE.write_text(modelfile, encoding="utf-8")
    rodar(["ollama", "create", MODELO_TREINADO, "-f", str(ARQ_MODELFILE)], "registro no Ollama")
    ARQ_GGUF.unlink()


# ======================================================================================================================
# ETAPA 8 — LINHA DE COMANDO (ponto de entrada)
#   Entrada: argumentos do terminal
#   Saída:   executa o comando pedido, que chama as etapas acima
#   Funções: 8.1 comando_indexar · 8.2 comando_perguntar · 8.3 comando_responder · 8.4 comando_treinar · 8.5 main
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 8.1 comando_indexar — python main.py indexar: etapas 1 a 3 (3.4)
#     Usada por: 8.5 main
# ----------------------------------------------------------------------------------------------------------------------
def comando_indexar(args) -> None:
    """Comando ``indexar``: lê o manual e cria o índice."""
    caminho = Path(args.manual)
    if not caminho.exists():
        raise ErroUsuario(f"manual não encontrado: {caminho} (coloque o manual.pdf ao lado do main.py)")
    print(f"Lendo {caminho} ...")
    inicio = time.perf_counter()
    n = indexar(caminho)
    print(f"{n} trechos indexados em {time.perf_counter() - inicio:.0f} s. Índice em {PASTA_INDICE}/")


# ----------------------------------------------------------------------------------------------------------------------
# 8.2 comando_perguntar — python main.py [perguntar]: etapas 3.6 -> 5.6 para cada pergunta
#     Usada por: 8.5 main
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# 8.3 comando_responder — python main.py responder: etapa 6
#     Usada por: 8.5 main
# ----------------------------------------------------------------------------------------------------------------------
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
    print("Depois rode: python avaliador.py")


# ----------------------------------------------------------------------------------------------------------------------
# 8.4 comando_treinar — python main.py treinar: etapa 7 (passos 7.2 a 7.7)
#     Usada por: 8.5 main
# ----------------------------------------------------------------------------------------------------------------------
def comando_treinar(args) -> None:
    """Comando ``treinar``: os cinco passos do ajuste fino, terminando com o ``qwen3-manual`` no Ollama."""
    # O mlx-lm só existe para Mac com Apple Silicon.
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        raise ErroUsuario("mlx-lm não instalado (só funciona em Mac com Apple Silicon)") from None
    # Confere o llama.cpp antes do treino, que é a parte demorada.
    for ferramenta in (CONVERSOR_LORA, EXPORT_LORA, QUANTIZE):
        if not ferramenta.is_file():
            raise ErroUsuario(f"{ferramenta.name} não encontrado em {ferramenta.parent} (veja o README)")
    # O pico de uso do disco é de ~6 GB (no passo 4).
    if shutil.disk_usage(RAIZ).free < 6 * 1024**3:
        print("aviso: menos de 6 GB livres no disco; a junção pode falhar", file=sys.stderr)
    inicio = time.perf_counter()
    indice = carregar_indice()
    print("Passo 1/5: preparando os dados ...", file=sys.stderr)
    n_treino = preparar_dados(indice)
    print(f"Passo 2/5: treinando com os {n_treino} trechos do manual ...", file=sys.stderr)
    treinar(n_treino, args.epocas)
    print("Passo 3/5: convertendo o adaptador ...", file=sys.stderr)
    converter_adaptador()
    print("Passo 4/5: somando o adaptador ao leitor e compactando ...", file=sys.stderr)
    juntar_no_leitor()
    print(f"Passo 5/5: registrando {MODELO_TREINADO} no Ollama ...", file=sys.stderr)
    registrar_no_ollama()
    minutos = (time.perf_counter() - inicio) / 60
    print(f"\nPronto em {minutos:.0f} min: o modelo {MODELO_TREINADO} está no Ollama.")
    print("Para comparar com o original: python avaliador.py")


# ----------------------------------------------------------------------------------------------------------------------
# 8.5 main — lê os argumentos e chama o comando; é onde a execução começa
#     Usada por: if __name__ abaixo
# ----------------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Lê os argumentos da linha de comando e executa o comando pedido.

    Entradas:
        argv: argumentos (padrão: os do terminal).
    Saídas:
        código de saída: 0 sucesso, 2 erro de uso, 130 interrompido.
    """
    parser = argparse.ArgumentParser(prog="main.py", description="Perguntas e respostas sobre o manual.")
    # Sem comando, abre o modo interativo (perguntar).
    parser.set_defaults(funcao=comando_perguntar, pergunta=None, modo="S3", modelo=MODELO_LEITOR)
    comandos = parser.add_subparsers(dest="comando")

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

    p = comandos.add_parser("treinar", help=f"ajuste fino (LoRA) do leitor com o manual; cria o {MODELO_TREINADO}")
    p.add_argument("--epocas", type=float, default=EPOCAS, help=f"passadas pelos trechos (padrão: {EPOCAS})")
    p.set_defaults(funcao=comando_treinar)

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
