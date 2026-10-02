"""Orquestração dos passos 1 a 4, manifesto do índice e carga dos índices para consulta."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import index_bm25, index_dense
from .chunking import Trecho, agrupar_trechos, estatisticas
from .ingest import construir_paragrafos
from .io_utils import (
    ErroUsuario,
    agora_iso,
    escrever_json,
    escrever_jsonl,
    ler_json,
    ler_jsonl,
    sha256_arquivo,
    versao_codigo,
)
from .ollama_client import OllamaClient

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)

VERSAO_MANIFESTO = 2
MIN_TEMAS = 3


def caminho_stats(cfg: Config) -> Path:
    """``data/trechos_stats.json`` (ao lado de ``paths.trechos``)."""
    return Path(cfg.paths.trechos).with_name("trechos_stats.json")


def caminho_manifesto(cfg: Config) -> Path:
    """``index/manifest.json``."""
    return Path(cfg.paths.index_dir) / "manifest.json"


def carregar_trechos(caminho: str | Path) -> list[Trecho]:
    """Lê ``trechos.jsonl``.

    Raises:
        ErroUsuario: se o arquivo não existir ou um registro estiver incompleto.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroUsuario(f"trechos não encontrados em {caminho}; rode `qa-manual indexar --manual <pdf>`")
    try:
        return [Trecho.de_dict(d) for d in ler_jsonl(caminho)]
    except KeyError as e:
        raise ErroUsuario(f"{caminho}: registro sem o campo {e}; rode `indexar` de novo.") from None


def escrever_manifesto(cfg: Config, pdf: Path, trechos: list[Trecho], n_paginas: int) -> dict:
    """Grava ``index/manifest.json`` (contrato 5.2)."""
    manifesto = {
        "versao_manifesto": VERSAO_MANIFESTO,
        "criado_em": agora_iso(),
        "manual_pdf": str(pdf),
        "hash_pdf_sha256": sha256_arquivo(pdf),
        "hash_trechos_sha256": sha256_arquivo(cfg.paths.trechos),
        "n_trechos": len(trechos),
        "n_paginas": n_paginas,
        "modelo_embeddings": cfg.modelos.embeddings,
        "dimensao": cfg.denso.dimensao,
        "normalizar_l2": cfg.denso.normalizar_l2,
        "params_trechos": asdict(cfg.trechos),
        "params_ingestao": asdict(cfg.ingestao),
        "bm25": asdict(cfg.bm25),
        "versao_codigo": versao_codigo(),
    }
    escrever_json(caminho_manifesto(cfg), manifesto)
    return manifesto


def _avisar_gold_invalido(cfg) -> None:
    """Reindexar com outros parâmetros muda os ids dos trechos e invalida a evidência do gold."""
    anterior = caminho_manifesto(cfg)
    if not anterior.is_file():
        return
    hash_antigo = ler_json(anterior).get("hash_trechos_sha256")
    if hash_antigo and hash_antigo != sha256_arquivo(cfg.paths.trechos):
        existentes = [str(p) for p in (cfg.paths.gold_dev, cfg.paths.gold_test) if Path(p).is_file()]
        if existentes:
            log.warning(
                "os trechos mudaram desde a última indexação; os ids de evidência em %s podem não "
                "corresponder mais aos trechos",
                ", ".join(existentes),
            )


def indexar(pdf: Path, cfg: Config, client: OllamaClient, reservar_ppl: int | None = None) -> dict:
    """Passos 1 a 4: PDF -> parágrafos -> ``trechos.jsonl`` -> índices BM25 e denso -> manifesto.

    Args:
        pdf: manual em PDF.
        cfg: configuração.
        client: cliente Ollama (embeddings).
        reservar_ppl: se informado, separa esse número de trechos em ``paths.ppl_holdout``.

    Returns:
        As estatísticas gravadas em ``trechos_stats.json``.
    """
    info: dict = {}
    paragrafos = construir_paragrafos(pdf, cfg, info)
    trechos = agrupar_trechos(paragrafos, cfg)
    if not trechos:
        raise ErroUsuario(f"Nenhum trecho gerado a partir de {pdf}.")

    escrever_jsonl(cfg.paths.trechos, [t.como_dict() for t in trechos])
    _avisar_gold_invalido(cfg)
    stats = estatisticas(trechos, cfg, info)
    escrever_json(caminho_stats(cfg), stats)
    log.info(
        "trechos: %d (%d temas), tokens mediana %s, máximo %s",
        stats["n_trechos"],
        stats["n_temas"],
        stats["tokens"]["mediana"],
        stats["tokens"]["maximo"],
    )
    if stats["n_temas"] < MIN_TEMAS:
        log.warning(
            "só %d tema(s) detectado(s); confira ingestao.regex_titulo para o estilo de títulos do manual",
            stats["n_temas"],
        )

    Path(cfg.paths.index_dir).mkdir(parents=True, exist_ok=True)
    index_bm25.construir(trechos, cfg)
    index_dense.construir(trechos, cfg, client)
    escrever_manifesto(cfg, pdf, trechos, info["n_paginas"])

    if reservar_ppl:
        from .gold import reservar_ppl as reservar

        gold_test = ler_jsonl(cfg.paths.gold_test) if Path(cfg.paths.gold_test).is_file() else None
        reservar(trechos, gold_test, reservar_ppl, cfg.gold.seed, Path(cfg.paths.ppl_holdout))
    return stats


def verificar_manifesto(cfg: Config) -> dict:
    """Recusa índices ausentes ou desatualizados em relação ao PDF, aos trechos e à configuração.

    Raises:
        ErroUsuario: com a instrução de rodar ``indexar``.
    """
    arq = caminho_manifesto(cfg)
    if not arq.is_file():
        raise ErroUsuario("índice não encontrado; rode `qa-manual indexar --manual <pdf>`")
    m = ler_json(arq)
    pdf = Path(m.get("manual_pdf", cfg.paths.manual_pdf))
    if not pdf.is_file():
        raise ErroUsuario(
            f"manual indexado não encontrado em {pdf}; não é possível conferir o índice. "
            "Rode `qa-manual indexar --manual <pdf>`."
        )
    motivos = []
    if sha256_arquivo(pdf) != m.get("hash_pdf_sha256"):
        motivos.append("o PDF mudou")
    if not Path(cfg.paths.trechos).is_file() or sha256_arquivo(cfg.paths.trechos) != m.get("hash_trechos_sha256"):
        motivos.append(f"{cfg.paths.trechos} mudou ou não existe")
    if m.get("modelo_embeddings") != cfg.modelos.embeddings or m.get("dimensao") != cfg.denso.dimensao:
        motivos.append(f"o índice usa {m.get('modelo_embeddings')} e a configuração pede {cfg.modelos.embeddings}")
    if m.get("bm25") != asdict(cfg.bm25):
        motivos.append("os parâmetros do BM25 mudaram")
    if motivos:
        raise ErroUsuario(f"índice desatualizado ({'; '.join(motivos)}); rode `qa-manual indexar --manual {pdf}`")
    return m


@dataclass
class Indices:
    """Trechos e índices carregados para os passos 8 a 11."""

    trechos: list[Trecho]
    bm25: index_bm25.IndiceBM25 | None
    denso: index_dense.IndiceDenso | None
    manifesto: dict

    def __post_init__(self) -> None:
        self.por_id = {t.id: t for t in self.trechos}


def carregar_indices(cfg: Config, client: OllamaClient, denso: bool = True) -> Indices:
    """Verifica o manifesto e carrega trechos, BM25 e (opcionalmente) o índice denso."""
    manifesto = verificar_manifesto(cfg)
    trechos = carregar_trechos(cfg.paths.trechos)
    bm25 = index_bm25.carregar(cfg)
    dens = index_dense.carregar(cfg, client) if denso else None
    if bm25.ids != [t.id for t in trechos] or (dens is not None and dens.ids != bm25.ids):
        raise ErroUsuario("índices e trechos estão inconsistentes; rode `qa-manual indexar --manual <pdf>`")
    return Indices(trechos=trechos, bm25=bm25, denso=dens, manifesto=manifesto)
