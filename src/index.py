"""Etapa 2: construção e carga dos índices BM25 (bm25s) e denso (FAISS + bge-m3 via Ollama)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import bm25s
import faiss
import numpy as np

from .ingest import ingerir
from .ollama_client import OllamaClient
from .utils import (
    ErroUsuario,
    escrever_json,
    escrever_jsonl,
    ler_json,
    ler_jsonl,
    lotes,
    sha256_arquivo,
    tokenizar_bm25,
)

VERSAO_INDICE = 1


def texto_indexavel(p: dict) -> str:
    """Texto usado nos índices: o título da seção ajuda tanto o BM25 quanto os embeddings."""
    return f"{p['secao']}\n{p['texto']}"


def normalizar_l2(m: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return (m / normas).astype("float32")


def construir_bm25(passagens: list[dict], destino: Path) -> None:
    corpus = [tokenizar_bm25(texto_indexavel(p)) for p in passagens]
    corpus = [toks or ["_vazio_"] for toks in corpus]
    modelo = bm25s.BM25()
    modelo.index(corpus, show_progress=False)
    destino.mkdir(parents=True, exist_ok=True)
    modelo.save(str(destino))


def embeddings(cliente: OllamaClient, modelo: str, textos: list[str], lote: int = 16, progresso: bool = False) -> np.ndarray:
    vetores: list[list[float]] = []
    for n, grupo in enumerate(lotes(textos, lote)):
        vetores.extend(cliente.embed(modelo, grupo))
        if progresso:
            feitos = min(len(textos), (n + 1) * lote)
            print(f"\r  embeddings: {feitos}/{len(textos)}", end="", file=sys.stderr, flush=True)
    if progresso:
        print(file=sys.stderr)
    return normalizar_l2(np.asarray(vetores, dtype="float32"))


def construir_denso(passagens: list[dict], destino_dir: Path, cliente: OllamaClient, modelo: str, lote: int) -> int:
    matriz = embeddings(cliente, modelo, [texto_indexavel(p) for p in passagens], lote, progresso=True)
    indice = faiss.IndexFlatIP(matriz.shape[1])
    indice.add(matriz)
    faiss.write_index(indice, str(destino_dir / "dense.faiss"))
    escrever_json(destino_dir / "dense_ids.json", [p["id"] for p in passagens])
    return int(matriz.shape[1])


def construir_indices(
    passagens: list[dict], estat: dict, manual: Path, params: dict, destino: Path, cliente: OllamaClient
) -> dict:
    destino.mkdir(parents=True, exist_ok=True)
    construir_bm25(passagens, destino / "bm25")
    dim = construir_denso(passagens, destino, cliente, params["embeddings"], params.get("lote_embeddings", 16))
    manifesto = {
        "versao": VERSAO_INDICE,
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "manual": str(manual),
        "manual_sha256": sha256_arquivo(manual),
        "modelo_embeddings": params["embeddings"],
        "dimensao_embeddings": dim,
        "segmentacao": {
            "tamanho_tokens": params["tamanho_tokens"],
            "sobreposicao_tokens": params["sobreposicao_tokens"],
            "palavras_por_token": params["palavras_por_token"],
            "detectar_tabelas": params.get("detectar_tabelas", True),
        },
        "passagens": len(passagens),
        "passagens_sha256": sha256_arquivo(destino / "passages.jsonl"),
        "bibliotecas": {
            "bm25s": getattr(bm25s, "__version__", "?"),
            "faiss": getattr(faiss, "__version__", "?"),
        },
    }
    escrever_json(destino / "manifest.json", manifesto)
    return manifesto


def indexar_manual(manual: Path, params: dict, destino: Path, cliente: OllamaClient) -> tuple[dict, dict]:
    """Etapas 1 e 2 em sequência: PDF → passages.jsonl + stats.json → índices BM25 e denso + manifest."""
    passagens, estat = ingerir(manual, params)
    destino.mkdir(parents=True, exist_ok=True)
    escrever_jsonl(destino / "passages.jsonl", passagens)
    escrever_json(destino / "stats.json", estat)
    manifesto = construir_indices(passagens, estat, manual, params, destino, cliente)
    return estat, manifesto


def indice_atualizado(destino: Path, manual: Path, params: dict) -> bool:
    arq = destino / "manifest.json"
    if not arq.exists():
        return False
    m = ler_json(arq)
    seg = m.get("segmentacao", {})
    return (
        m.get("manual_sha256") == sha256_arquivo(manual)
        and m.get("modelo_embeddings") == params["embeddings"]
        and seg.get("tamanho_tokens") == params["tamanho_tokens"]
        and seg.get("sobreposicao_tokens") == params["sobreposicao_tokens"]
        and seg.get("palavras_por_token") == params["palavras_por_token"]
    )


def verificar_manifesto(dir_indice: Path, modelo_embeddings: str | None = None) -> dict:
    """Garante que o índice existe, não está desatualizado e usa o modelo de embeddings esperado."""
    arq = dir_indice / "manifest.json"
    if not arq.exists():
        raise ErroUsuario(
            f"Índice não encontrado em {dir_indice}/. Rode antes: python -m src.cli indexar --manual <arquivo.pdf>"
        )
    manifesto = ler_json(arq)
    manual = Path(manifesto.get("manual", ""))
    if manual.exists():
        if sha256_arquivo(manual) != manifesto.get("manual_sha256"):
            raise ErroUsuario(
                f"O manual {manual} mudou desde a indexação. Rode novamente: python -m src.cli indexar --manual {manual}"
            )
    else:
        print(f"Aviso: manual {manual} não encontrado; não foi possível verificar se o índice está atualizado.",
              file=sys.stderr)
    if modelo_embeddings and manifesto.get("modelo_embeddings") != modelo_embeddings:
        raise ErroUsuario(
            f"O índice usa embeddings '{manifesto.get('modelo_embeddings')}', mas a configuração pede "
            f"'{modelo_embeddings}'. Reindexe ou ajuste a configuração."
        )
    return manifesto


class Indice:
    """Passagens + índices carregados em memória."""

    def __init__(self, dir_indice: Path, carregar_denso: bool = True):
        self.dir = Path(dir_indice)
        self.manifesto = verificar_manifesto(self.dir)
        self.passagens = ler_jsonl(self.dir / "passages.jsonl")
        self.por_id = {p["id"]: p for p in self.passagens}
        self.bm25 = bm25s.BM25.load(str(self.dir / "bm25"))
        self.denso = None
        self.ids_denso: list[str] = []
        if carregar_denso and (self.dir / "dense.faiss").exists():
            self.denso = faiss.read_index(str(self.dir / "dense.faiss"))
            self.ids_denso = json.loads((self.dir / "dense_ids.json").read_text(encoding="utf-8"))
            if len(self.ids_denso) != self.denso.ntotal:
                raise ErroUsuario("dense_ids.json e dense.faiss estão inconsistentes; reindexe.")

    @property
    def modelo_embeddings(self) -> str:
        return self.manifesto["modelo_embeddings"]
