"""Passo 15: executa a grade (config x leitor x pergunta), agrega as métricas e gera tabelas e gráficos.

``respostas.jsonl`` é gravado registro a registro; ``--retomar <run_id>`` pula os pares
(config, leitor, q_id, repetição) já gravados.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import metrics
from .gold import validar_gold
from .indexacao import carregar_indices
from .io_utils import (
    ErroUsuario,
    agora_iso,
    anexar_jsonl,
    anexar_log_arquivo,
    criar_run_dir,
    escrever_json,
    ler_json,
    ler_jsonl,
    novo_run_id,
    remover_log_arquivo,
    sha256_arquivo,
    sha256_texto,
    versao_codigo,
)
from .ollama_client import ModeloAusente, OllamaClient, instrucao_modelo, modelo_presente
from .pipeline import perguntar

log = logging.getLogger(__name__)

# Paleta categórica validada (slots 1 e 2) e tintas neutras para os gráficos estáticos.
COR_LEITOR = {"base": "#2a78d6", "ajustado": "#eb6834"}
COR_SUPERFICIE = "#fcfcfb"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e4e3df"

BertscoreFn = Callable[[list[str], list[str], str, int], list[float]]


def _chave(reg: dict) -> tuple:
    return reg["config"], reg["leitor"], reg["q_id"], reg.get("repeticao", 1)


def verificar_modelos(client: OllamaClient, modelos: list[str]) -> None:
    """Garante que os modelos estão instalados no Ollama antes de começar."""
    disponiveis = client.modelos_disponiveis()
    for m in modelos:
        if not modelo_presente(m, disponiveis):
            raise ModeloAusente(instrucao_modelo(m))


def aquecer(client: OllamaClient, modelo: str, cfg) -> None:
    """Carrega o modelo na memória antes de medir latência (a primeira chamada inclui o carregamento)."""
    opcoes = cfg.decodificacao.opcoes() | {"num_predict": 1}
    client.chat(
        model=modelo, messages=[{"role": "user", "content": "Olá"}], options=opcoes, think=cfg.decodificacao.think
    )


# ---------------------------------------------------------------------------
# BERTScore com cache em disco
# ---------------------------------------------------------------------------


def bertscore_com_cache(
    preds: list[str], refs: list[str], modelo: str, num_layers: int, cache: Path, fn: BertscoreFn | None = None
) -> list[float]:
    """BERTScore F1 calculado em um único lote, reaproveitando pares já calculados.

    O cache guarda só hashes de ``(modelo, camadas, pred, ref)`` e o valor, nunca o texto.
    """
    fn = fn or metrics.bertscore_f1
    armazenado: dict[str, float] = ler_json(cache) if cache.is_file() else {}
    chaves = [sha256_texto(f"{modelo}|{num_layers}|{p}|{r}") for p, r in zip(preds, refs, strict=True)]
    faltando = sorted({i for i, c in enumerate(chaves) if c not in armazenado}, key=lambda i: chaves[i])
    unicos: dict[str, int] = {}
    for i in faltando:
        unicos.setdefault(chaves[i], i)
    if unicos:
        idx = list(unicos.values())
        log.info("BERTScore: %d pares novos (%d em cache)", len(idx), len(chaves) - len(faltando))
        valores = fn([preds[i] for i in idx], [refs[i] for i in idx], modelo, num_layers)
        armazenado.update({chaves[i]: v for i, v in zip(idx, valores, strict=True)})
        escrever_json(cache, armazenado)
    return [armazenado[c] for c in chaves]


# ---------------------------------------------------------------------------
# Métricas e saídas
# ---------------------------------------------------------------------------


def localizar_perplexidade(runs_dir: Path, explicito: Path | None) -> Path | None:
    """``perplexidade.json`` indicado ou, se omitido, o mais recente em ``runs/``."""
    if explicito is not None:
        if not Path(explicito).is_file():
            raise ErroUsuario(f"Arquivo não encontrado: {explicito}")
        return Path(explicito)
    candidatos = sorted(Path(runs_dir).glob("*/perplexidade.json"))
    return candidatos[-1] if candidatos else None


def calcular_metricas(
    registros: list[dict],
    gold: dict[str, dict],
    cfg,
    run_dir: Path,
    bertscore_fn: BertscoreFn | None = None,
    perplexidade: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pontua os registros, calcula BERTScore em lote, agrega por (config, leitor) e o ganho pareado.

    Returns:
        ``(metrics, pareado)`` como DataFrames.
    """
    pontuados = [metrics.pontuar(r, gold[r["q_id"]]) for r in registros if r["q_id"] in gold]
    alvo = [i for i, p in enumerate(pontuados) if p["respondivel"] and not p["absteve"]]
    if alvo:
        av = cfg.avaliacao
        cache = Path(cfg.paths.runs_dir) / ".cache" / "bertscore.json"
        valores = bertscore_com_cache(
            [pontuados[i]["pred"] for i in alvo],
            [pontuados[i]["ref"] for i in alvo],
            av.bertscore_modelo,
            av.bertscore_num_layers,
            cache,
            bertscore_fn,
        )
        for i, v in zip(alvo, valores, strict=True):
            pontuados[i]["bertscore_f1"] = v
    df = pd.DataFrame(metrics.agregar(pontuados, cfg.k))
    if perplexidade and not df.empty:
        df["perplexidade"] = df["leitor"].map(lambda lt: perplexidade.get(lt, math.nan))
    pareado = pd.DataFrame(metrics.ganho_pareado(pontuados, seed=cfg.decodificacao.seed))
    df.to_csv(run_dir / "metrics.csv", index=False)
    pareado.to_csv(run_dir / "pareado.csv", index=False)
    return df, pareado


def _estilo_eixo(ax) -> None:
    ax.set_facecolor(COR_SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COR_GRADE)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=9, length=0)
    ax.yaxis.grid(True, color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def gerar_graficos(df: pd.DataFrame, run_dir: Path, k: int = 5) -> list[Path]:
    """Barras de EM/F1/BERTScore por configuração (séries base e ajustado) e de Recall@k por configuração."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return []
    configs = list(dict.fromkeys(df["config"]))
    leitores = [lt for lt in ("base", "ajustado") if lt in set(df["leitor"])]
    largura = 0.8 / max(1, len(leitores))
    saidas = []

    fig, eixos = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=COR_SUPERFICIE)
    for ax, (coluna, titulo) in zip(
        eixos, [("em", "Exact Match"), ("f1_tokens", "F1 de tokens"), ("bertscore_f1", "BERTScore F1")], strict=True
    ):
        _estilo_eixo(ax)
        for j, leitor in enumerate(leitores):
            sub = df[df["leitor"] == leitor].set_index("config").reindex(configs)
            xs = [i + (j - (len(leitores) - 1) / 2) * largura for i in range(len(configs))]
            ax.bar(
                xs,
                sub[coluna].fillna(0),
                width=largura,
                color=COR_LEITOR[leitor],
                label=leitor,
                edgecolor=COR_SUPERFICIE,
                linewidth=2,
            )
        ax.set_xticks(range(len(configs)), configs)
        ax.set_title(titulo, color=COR_TEXTO, fontsize=11, loc="left")
        ax.set_ylim(0, 1)
    eixos[0].set_ylabel("média nos itens respondíveis", color=COR_TEXTO_SECUNDARIO, fontsize=9)
    eixos[0].legend(frameon=False, fontsize=9, labelcolor=COR_TEXTO_SECUNDARIO, title="leitor", title_fontsize=9)
    fig.tight_layout()
    caminho = run_dir / "respostas_por_config.png"
    fig.savefig(caminho, dpi=150, facecolor=COR_SUPERFICIE)
    plt.close(fig)
    saidas.append(caminho)

    coluna = f"recall_at_{k}"
    recall = df.groupby("config", sort=False)[coluna].mean().reindex(configs).dropna()
    if not recall.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor=COR_SUPERFICIE)
        _estilo_eixo(ax)
        ax.bar(range(len(recall)), recall.values, width=0.6, color=COR_LEITOR["base"])
        ax.set_xticks(range(len(recall)), list(recall.index))
        ax.set_ylim(0, 1)
        ax.set_title(f"Recall@{k} por configuração", color=COR_TEXTO, fontsize=11, loc="left")
        for i, v in enumerate(recall.values):
            ax.annotate(
                f"{v:.2f}",
                (i, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COR_TEXTO_SECUNDARIO,
            )
        fig.tight_layout()
        caminho = run_dir / "recall_por_config.png"
        fig.savefig(caminho, dpi=150, facecolor=COR_SUPERFICIE)
        plt.close(fig)
        saidas.append(caminho)
    return saidas


def tabela_texto(df: pd.DataFrame) -> str:
    """Tabela para o terminal (Markdown se ``tabulate`` estiver instalado)."""
    arredondado = df.round(3)
    try:
        return arredondado.to_markdown(index=False)
    except ImportError:
        return arredondado.to_string(index=False)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------


def avaliar(
    gold_path: Path,
    configs: list,
    leitores: list[str],
    repeticoes: int,
    client: OllamaClient,
    retomar: str | None = None,
    perplexidade_path: Path | None = None,
    limite: int | None = None,
    bertscore_fn: BertscoreFn | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Executa a avaliação completa.

    Args:
        gold_path: arquivo gold (``gold_dev.jsonl`` ou ``gold_test.jsonl``).
        configs: configurações experimentais carregadas (uma por S0..S3), com o mesmo ``base``.
        leitores: ``base`` e/ou ``ajustado``.
        repeticoes: repetições de cada pergunta.
        client: cliente Ollama.
        retomar: ``run_id`` de uma execução interrompida a continuar.
        perplexidade_path: ``perplexidade.json``; se omitido, usa o mais recente em ``runs/``, se houver.
        limite: avalia só as primeiras ``limite`` perguntas (experimento mínimo).
        bertscore_fn: substitui o cálculo do BERTScore (testes).

    Returns:
        ``(run_dir, metrics)``.
    """
    cfg = configs[0]
    gold_lista = ler_jsonl(gold_path)
    validar_gold(gold_lista, gold_path)
    if limite:
        gold_lista = gold_lista[:limite]
    gold = {it["id"]: it for it in gold_lista}
    hash_gold = sha256_arquivo(gold_path)

    run_id = retomar or novo_run_id("avaliar")
    run_dir = Path(cfg.paths.runs_dir) / run_id
    descricao = {
        "run_id": run_id,
        "criado_em": agora_iso(),
        "versao_codigo": versao_codigo(),
        "gold": str(gold_path),
        "gold_sha256": hash_gold,
        "n_perguntas": len(gold_lista),
        "leitores": leitores,
        "repeticoes": repeticoes,
        "limite": limite,
        "configs": {c.nome: c.como_dict() for c in configs},
    }
    if retomar:
        if not (run_dir / "config.json").is_file():
            raise ErroUsuario(f"execução {retomar} não encontrada em {cfg.paths.runs_dir}/")
        anterior = ler_json(run_dir / "config.json")
        if anterior.get("gold_sha256") != hash_gold:
            raise ErroUsuario(f"o gold {gold_path} difere do usado em {retomar}; não é possível retomar.")
        for campo in ("configs", "leitores", "repeticoes", "limite"):
            if anterior.get(campo) != json.loads(json.dumps(descricao[campo])):
                raise ErroUsuario(f"'{campo}' difere do usado em {retomar}; retome com os mesmos argumentos.")
    criar_run_dir(cfg.paths.runs_dir, run_id)
    handler = anexar_log_arquivo(run_dir / "log.txt")
    try:
        if not retomar:
            escrever_json(run_dir / "config.json", descricao)
        respostas = run_dir / "respostas.jsonl"
        feitos = {_chave(r) for r in ler_jsonl(respostas)} if respostas.is_file() else set()
        if feitos:
            log.info("retomando %s: %d registros já gravados", run_id, len(feitos))

        precisa_indice = any(c.retriever != "nenhum" for c in configs)
        precisa_denso = any(c.retriever in ("denso", "hibrido") for c in configs)
        indices = carregar_indices(cfg, client, denso=precisa_denso) if precisa_indice else None
        modelos = [cfg.modelo_leitor(lt) for lt in leitores]
        verificar_modelos(client, modelos + ([cfg.modelos.embeddings] if precisa_denso else []))

        total = len(leitores) * len(configs) * repeticoes * len(gold_lista)
        barra = tqdm(total=total, desc="avaliação", unit="resposta", initial=len(feitos))
        # Leitor por fora: evita trocar de modelo na memória a cada pergunta.
        for leitor in leitores:
            pendentes = [
                (c, rep, it)
                for c in configs
                for rep in range(1, repeticoes + 1)
                for it in gold_lista
                if (c.nome, leitor, it["id"], rep) not in feitos
            ]
            if not pendentes:
                continue
            if precisa_denso:
                client.embed(cfg.modelos.embeddings, ["aquecimento"])
            aquecer(client, cfg.modelo_leitor(leitor), cfg)
            for c, rep, it in pendentes:
                reg = perguntar(
                    it["pergunta"], c, leitor, indices, client, q_id=it["id"], tipo=it["tipo"], run_id=run_id
                )
                reg["repeticao"] = rep
                anexar_jsonl(respostas, reg)
                barra.update(1)
        barra.close()

        registros = [r for r in ler_jsonl(respostas) if r["q_id"] in gold]
        ppl_arq = localizar_perplexidade(cfg.paths.runs_dir, perplexidade_path)
        ppl = None
        if ppl_arq:
            log.info("perplexidade de %s", ppl_arq)
            ppl = {k: v for k, v in ler_json(ppl_arq).get("perplexidade", {}).items()}
        df, _ = calcular_metricas(registros, gold, cfg, run_dir, bertscore_fn, ppl)
        gerar_graficos(df, run_dir, cfg.k)
        log.info("métricas gravadas em %s", run_dir / "metrics.csv")
        return run_dir, df
    except KeyboardInterrupt:
        log.warning("avaliação interrompida; retome com `qa-manual avaliar --gold %s --retomar %s`", gold_path, run_id)
        raise
    finally:
        remover_log_arquivo(handler)


def resumo_json(df: pd.DataFrame) -> str:
    """Métricas como JSON compacto (para logs e testes)."""
    return json.dumps(df.round(4).to_dict(orient="records"), ensure_ascii=False)
