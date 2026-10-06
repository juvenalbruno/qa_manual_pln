"""Carregamento e validação da configuração YAML (``configs/``) em dataclasses imutáveis.

``carregar(base, *sobrescritas)`` lê ``base.yaml`` e aplica as sobrescritas em ordem. Seções aninhadas são
mescladas chave a chave, de modo que um YAML pode alterar só ``trechos.max_tokens`` sem repetir a seção
inteira. Chaves desconhecidas, tipos errados e valores fora do domínio encerram com :class:`ErroUsuario`.
"""

from __future__ import annotations

import dataclasses
import os
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io_utils import RAIZ, ErroUsuario

DIR_CONFIGS = RAIZ / "configs"
BASE_PADRAO = "configs/base.yaml"
MODOS_RETRIEVER = ("nenhum", "bm25", "denso", "hibrido")
LEITORES = ("base", "ajustado")


@dataclass(frozen=True)
class Paths:
    manual_pdf: Path
    trechos: Path
    index_dir: Path
    runs_dir: Path
    gold_dev: Path
    gold_test: Path
    ppl_holdout: Path


@dataclass(frozen=True)
class OllamaCfg:
    host: str
    timeout_s: float


@dataclass(frozen=True)
class Modelos:
    leitor_base: str
    leitor_ajustado: str
    embeddings: str
    gerador: str


@dataclass(frozen=True)
class Ingestao:
    remover_cabecalhos_rodapes: bool
    limiar_repeticao_cabecalho: float
    regex_titulo: str
    validar_sequencia_titulos: bool
    min_chars_paragrafo: int


@dataclass(frozen=True)
class TrechosCfg:
    max_tokens: int
    min_tokens: int
    tokens_por_palavra: float
    sobreposicao_paragrafos: int


@dataclass(frozen=True)
class BM25Cfg:
    k1: float
    b: float
    remover_stopwords: bool
    manter_acentos: bool


@dataclass(frozen=True)
class DensoCfg:
    dimensao: int
    normalizar_l2: bool
    batch_size: int


@dataclass(frozen=True)
class Decodificacao:
    temperature: float
    top_p: float
    seed: int
    num_ctx: int
    num_predict: int
    think: bool

    def opcoes(self) -> dict:
        """Opções de amostragem no formato do Ollama."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


@dataclass(frozen=True)
class Abstencao:
    frase: str
    limiar_score: float | None


@dataclass(frozen=True)
class GoldCfg:
    n_candidatas: int
    frac_sem_resposta: float
    frac_dev: float
    seed: int


@dataclass(frozen=True)
class Avaliacao:
    configs: list[str]
    leitores: list[str]
    repeticoes: int
    bertscore_modelo: str
    bertscore_num_layers: int
    ppl_ctx: int


@dataclass(frozen=True)
class Config:
    """Configuração efetiva de uma execução. ``nome`` identifica a configuração experimental (S0 a S3)."""

    paths: Paths
    ollama: OllamaCfg
    modelos: Modelos
    ingestao: Ingestao
    trechos: TrechosCfg
    bm25: BM25Cfg
    denso: DensoCfg
    retriever: str
    k: int
    k_candidatos_por_lista: int
    rrf_c: int
    decodificacao: Decodificacao
    abstencao: Abstencao
    gold: GoldCfg
    avaliacao: Avaliacao
    nome: str = "base"

    def modelo_leitor(self, leitor: str) -> str:
        """Nome do modelo no Ollama para o leitor ``base`` ou ``ajustado``."""
        if leitor == "base":
            return self.modelos.leitor_base
        if leitor == "ajustado":
            return self.modelos.leitor_ajustado
        raise ErroUsuario(f"Leitor desconhecido: {leitor}. Use: {', '.join(LEITORES)}.")

    def como_dict(self) -> dict:
        """Configuração completa como dicionário serializável (caminhos viram texto)."""
        return _serializavel(dataclasses.asdict(self))

    def com(self, **mudancas: Any) -> Config:
        """Cópia com campos de primeiro nível alterados, por exemplo ``cfg.com(k=8)``."""
        return dataclasses.replace(self, **mudancas)


# Campos preenchidos pelo carregador, não pelo YAML.
_CAMPOS_INTERNOS = {"nome"}


def _serializavel(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {k: _serializavel(v) for k, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [_serializavel(v) for v in valor]
    if isinstance(valor, Path):
        return str(valor)
    return valor


def resolver_config(ref: str | Path) -> Path:
    """Localiza um YAML de configuração.

    Aceita um nome curto (``S3`` vira ``configs/S3.yaml``) ou um caminho. Caminhos relativos são procurados
    primeiro no diretório atual e depois na raiz do repositório.
    """
    ref = Path(ref)
    candidatos = [DIR_CONFIGS / f"{ref}.yaml"] if not ref.suffix else [ref, RAIZ / ref]
    for c in candidatos:
        if c.is_file():
            return c
    disponiveis = ", ".join(sorted(p.stem for p in DIR_CONFIGS.glob("*.yaml")))
    raise ErroUsuario(f"Configuração não encontrada: {ref}. Disponíveis em configs/: {disponiveis}.")


def _ler_yaml(caminho: Path) -> dict:
    try:
        dados = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ErroUsuario(f"YAML inválido em {caminho}: {e}") from None
    if not isinstance(dados, dict):
        raise ErroUsuario(f"{caminho}: o YAML deve ser um mapeamento de chaves.")
    return dados


def mesclar(base: dict, sobrescrita: dict) -> dict:
    """Mescla ``sobrescrita`` sobre ``base``; seções (dicionários) são mescladas recursivamente."""
    saida = dict(base)
    for chave, valor in sobrescrita.items():
        if isinstance(valor, dict) and isinstance(saida.get(chave), dict):
            saida[chave] = mesclar(saida[chave], valor)
        else:
            saida[chave] = valor
    return saida


def _converter(valor: Any, tipo: Any, caminho: str) -> Any:
    """Valida ``valor`` contra a anotação ``tipo`` e converte quando necessário."""
    origem = typing.get_origin(tipo)
    if origem in (typing.Union, types.UnionType):
        opcoes = typing.get_args(tipo)
        if valor is None and type(None) in opcoes:
            return None
        tipo = next(t for t in opcoes if t is not type(None))
        origem = typing.get_origin(tipo)
    if dataclasses.is_dataclass(tipo):
        if not isinstance(valor, dict):
            raise ErroUsuario(f"Configuração: '{caminho}' deve ser uma seção (mapeamento).")
        return _construir(tipo, valor, caminho)
    if origem is list:
        (tipo_item,) = typing.get_args(tipo)
        if not isinstance(valor, list):
            raise ErroUsuario(f"Configuração: '{caminho}' deve ser uma lista.")
        return [_converter(v, tipo_item, f"{caminho}[{i}]") for i, v in enumerate(valor)]
    if tipo is Path:
        if not isinstance(valor, str) or not valor:
            raise ErroUsuario(f"Configuração: '{caminho}' deve ser um caminho (texto).")
        return Path(valor)
    if tipo is bool:
        if not isinstance(valor, bool):
            raise ErroUsuario(f"Configuração: '{caminho}' deve ser true ou false.")
        return valor
    if tipo is int:
        if isinstance(valor, bool) or not isinstance(valor, int):
            raise ErroUsuario(f"Configuração: '{caminho}' deve ser um número inteiro.")
        return valor
    if tipo is float:
        if isinstance(valor, bool) or not isinstance(valor, int | float):
            raise ErroUsuario(f"Configuração: '{caminho}' deve ser um número.")
        return float(valor)
    if tipo is str:
        if not isinstance(valor, str):
            raise ErroUsuario(f"Configuração: '{caminho}' deve ser texto.")
        return valor
    raise TypeError(f"Tipo de configuração não suportado: {tipo}")  # pragma: no cover


def _construir(cls: type, dados: dict, prefixo: str = "") -> Any:
    dicas = typing.get_type_hints(cls)
    campos = [f for f in dataclasses.fields(cls) if f.name not in _CAMPOS_INTERNOS]
    nomes = {f.name for f in campos}
    desconhecidas = sorted(set(dados) - nomes)
    if desconhecidas:
        lista = ", ".join(f"{prefixo}{k}" for k in desconhecidas)
        raise ErroUsuario(f"Configuração: chave(s) desconhecida(s): {lista}.")
    faltando = sorted(n for n in nomes if n not in dados)
    if faltando:
        lista = ", ".join(f"{prefixo}{k}" for k in faltando)
        raise ErroUsuario(f"Configuração: chave(s) obrigatória(s) ausente(s): {lista}.")
    valores = {f.name: _converter(dados[f.name], dicas[f.name], f"{prefixo}{f.name}") for f in campos}
    return cls(**valores)


def _validar_dominio(cfg: Config) -> None:
    erros = []
    if cfg.retriever not in MODOS_RETRIEVER:
        erros.append(f"retriever deve ser um de {', '.join(MODOS_RETRIEVER)}")
    if cfg.k < 1:
        erros.append("k deve ser >= 1")
    if cfg.k_candidatos_por_lista < cfg.k:
        erros.append("k_candidatos_por_lista deve ser >= k")
    if cfg.rrf_c < 0:
        erros.append("rrf_c deve ser >= 0")
    if not 0 < cfg.ingestao.limiar_repeticao_cabecalho <= 1:
        erros.append("ingestao.limiar_repeticao_cabecalho deve estar em (0, 1]")
    try:
        re.compile(cfg.ingestao.regex_titulo)
    except re.error as e:
        erros.append(f"ingestao.regex_titulo inválida ({e})")
    t = cfg.trechos
    if t.max_tokens < 1 or t.min_tokens < 0 or t.min_tokens > t.max_tokens:
        erros.append("trechos: exige 0 <= min_tokens <= max_tokens e max_tokens >= 1")
    if t.tokens_por_palavra <= 0:
        erros.append("trechos.tokens_por_palavra deve ser > 0")
    if t.sobreposicao_paragrafos < 0:
        erros.append("trechos.sobreposicao_paragrafos deve ser >= 0")
    if cfg.denso.dimensao < 1 or cfg.denso.batch_size < 1:
        erros.append("denso.dimensao e denso.batch_size devem ser >= 1")
    if not 0 <= cfg.gold.frac_sem_resposta < 1 or not 0 < cfg.gold.frac_dev < 1:
        erros.append("gold.frac_sem_resposta deve estar em [0, 1) e gold.frac_dev em (0, 1)")
    if cfg.gold.n_candidatas < 1:
        erros.append("gold.n_candidatas deve ser >= 1")
    invalidos = [lt for lt in cfg.avaliacao.leitores if lt not in LEITORES]
    if invalidos:
        erros.append(f"avaliacao.leitores aceita apenas {', '.join(LEITORES)}")
    if cfg.avaliacao.repeticoes < 1:
        erros.append("avaliacao.repeticoes deve ser >= 1")
    if not cfg.abstencao.frase.strip():
        erros.append("abstencao.frase não pode ser vazia")
    if erros:
        raise ErroUsuario("Configuração inválida:\n  - " + "\n  - ".join(erros))


def carregar(base: str | Path = BASE_PADRAO, *sobrescritas: str | Path | None, nome: str | None = None) -> Config:
    """Carrega ``base`` e aplica as sobrescritas em ordem.

    Args:
        base: caminho do YAML base (padrão ``configs/base.yaml``).
        *sobrescritas: YAMLs ou nomes curtos (``S3``) aplicados em sequência; ``None`` é ignorado.
        nome: nome da configuração experimental; por padrão, o nome do último YAML de sobrescrita ou ``base``.

    Returns:
        A configuração validada. A variável de ambiente ``OLLAMA_HOST``, se definida, substitui
        ``ollama.host`` (o cliente continua aceitando só endereços locais).

    Raises:
        ErroUsuario: se algum arquivo não existir ou a configuração for inválida.
    """
    dados = _ler_yaml(resolver_config(base))
    ultimo = None
    for s in sobrescritas:
        if s is None:
            continue
        caminho = resolver_config(s)
        dados = mesclar(dados, _ler_yaml(caminho))
        ultimo = caminho.stem
    host = os.environ.get("OLLAMA_HOST")
    if host:
        dados = mesclar(dados, {"ollama": {"host": host if "://" in host else f"http://{host}"}})
    cfg = _construir(Config, dados)
    cfg = dataclasses.replace(cfg, nome=nome or ultimo or "base")
    _validar_dominio(cfg)
    return cfg


def carregar_experimento(exp: str, base: str | Path = BASE_PADRAO, sobrescrever: str | Path | None = None) -> Config:
    """Configuração de um experimento (``S0`` a ``S3``).

    Ordem de aplicação: base, depois ``sobrescrever``, depois ``configs/<exp>.yaml``.
    """
    return carregar(base, sobrescrever, exp, nome=Path(exp).stem)
