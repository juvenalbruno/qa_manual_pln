"""Fixtures comuns: PDF sintético (texto inventado), cliente Ollama falso e índices construídos sobre o PDF."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytest

from qa_manual import config, indexacao
from qa_manual.ollama_client import OllamaFake

CABECALHO = "Manual Fictício de Operações - Terminal Exemplo"
CSS = (
    "body { font-family: sans-serif; font-size: 10pt; } "
    "h1 { font-size: 15pt; font-weight: bold; } p { margin-bottom: 8pt; }"
)

PALAVRAS_LONGAS = (
    "O plano de contingência do terminal descreve responsabilidades das equipes de operação, manutenção, "
    "segurança e atendimento, com rotinas de inspeção, comunicação com a autoridade portuária, registro de "
    "ocorrências e treinamento periódico dos colaboradores envolvidos nas manobras."
).split()
PARAGRAFO_700 = " ".join(PALAVRAS_LONGAS[i % len(PALAVRAS_LONGAS)] for i in range(700)) + "."

# Cada página: lista de (tag, texto). O texto é fictício e não vem de nenhum manual real.
PAGINAS = [
    [
        ("h1", "1 Introdução"),
        (
            "p",
            "Este manual fictício descreve as rotinas do Terminal Exemplo e serve apenas para testes automatizados "
            "do sistema de perguntas e respostas, sem qualquer relação com documentos reais de empresas.",
        ),
        (
            "p",
            "As equipes do terminal trabalham em três turnos de oito horas, e cada turno tem um supervisor de "
            "operações responsável por aprovar as manobras, registrar as ocorrências no sistema interno e "
            "comunicar a coordenação sobre atrasos, avarias ou condições climáticas adversas no cais.",
        ),
    ],
    [
        (
            "p",
            "O manual é revisado todos os anos pela coordenação de operações. A libera-<br>ção de novas versões "
            "depende da aprovação do comitê de qualidade, que se reúne na primeira semana de cada trimestre "
            "para avaliar sugestões enviadas pelo formulário interno de melhorias.",
        ),
    ],
    [
        ("h1", "2.1 Atracação"),
        (
            "p",
            "O agente marítimo deve confirmar a chegada do navio com antecedência mínima de 72 horas, informando "
            "o calado, o comprimento total e a quantidade de contêineres a descarregar. Pedidos recebidos após "
            "esse prazo entram no fim da fila de programação do terminal.",
        ),
        (
            "p",
            "Navios com mais de 200 metros de comprimento exigem dois rebocadores durante a manobra, e a "
            "atracação noturna só é autorizada com visibilidade superior a duas milhas náuticas.",
        ),
    ],
    [
        ("p", PARAGRAFO_700),
    ],
    [
        ("h1", "2.2 Liberação de carga"),
        (
            "p",
            "A liberação da carga ocorre em até 48 horas após a atracação, desde que todos os tributos estejam "
            "recolhidos e não haja exigência de inspeção pela fiscalização aduaneira do porto.",
        ),
        (
            "p",
            "O importador acompanha o andamento pelo portal do terminal usando o número do conhecimento de "
            "embarque. A armazenagem é gratuita nos primeiros sete dias corridos contados da descarga.",
        ),
    ],
    [
        (
            "p",
            "Para retirar a carga, o transportador agenda a janela de retirada com antecedência mínima de 24 "
            "horas e apresenta a ordem de coleta na portaria principal no dia marcado.",
        ),
        ("p", "Divergências de lacre devem ser registradas no termo de avaria antes da saída do veículo."),
    ],
]


def gerar_pdf_sintetico(destino: Path) -> Path:
    """Gera o PDF de 6 páginas com cabeçalho e rodapé repetidos, títulos numerados e hifenização."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for n, conteudo in enumerate(PAGINAS, start=1):
        pg = doc.new_page(width=595, height=842)
        corpo = "".join(f"<{tag}>{texto}</{tag}>" for tag, texto in conteudo)
        tamanho = "font-size: 7pt;" if len(corpo) > 3000 else ""
        pg.insert_htmlbox(pymupdf.Rect(60, 80, 535, 780), corpo, css=CSS + f" p {{ {tamanho} }}")
        pg.insert_text((60, 40), CABECALHO, fontsize=8)
        pg.insert_text((260, 815), f"Página {n} de {len(PAGINAS)}", fontsize=8)
    doc.save(destino)
    doc.close()
    return destino


_TRECHO_PROMPT = re.compile(r"^\[(\d+)\] \(seção (.+?), p\. (\d+)\) (.*)$", re.MULTILINE)
_PALAVRA = re.compile(r"[^\W_]{3,}")


def leitor_extrativo(modelo: str, prompt: str) -> str:
    """Leitor falso: copia a frase do trecho com maior sobreposição com a pergunta e cita a fonte."""
    pergunta = prompt.rsplit("Pergunta:", 1)[-1].split("Resposta:", 1)[0].lower()
    trechos = _TRECHO_PROMPT.findall(prompt)
    if not trechos:
        return "Não encontrado no manual."
    termos = set(_PALAVRA.findall(pergunta))
    melhor, pontos = None, 0
    for _, secao, pagina, texto in trechos:
        for frase in re.split(r"(?<=[.!?])\s+", texto):
            p = len(termos & set(_PALAVRA.findall(frase.lower())))
            if p > pontos:
                melhor, pontos = (frase.rstrip("."), secao, pagina), p
    if melhor is None or pontos < 2:
        return "Não encontrado no manual."
    return f"{melhor[0]} (seção {melhor[1]}, p. {melhor[2]})."


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> config.Config:
    """Configuração padrão com o diretório de trabalho isolado em ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return config.carregar()


@pytest.fixture
def pdf_sintetico(cfg) -> Path:
    return gerar_pdf_sintetico(Path(cfg.paths.manual_pdf))


@pytest.fixture
def client_fake() -> OllamaFake:
    return OllamaFake(resposta=leitor_extrativo)


@pytest.fixture
def indices(cfg, pdf_sintetico, client_fake) -> indexacao.Indices:
    """Trechos e índices BM25/denso construídos sobre o PDF sintético com embeddings falsos."""
    indexacao.indexar(pdf_sintetico, cfg, client_fake)
    return indexacao.carregar_indices(cfg, client_fake)
