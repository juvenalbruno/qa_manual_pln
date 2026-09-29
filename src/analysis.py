"""Etapa 7: comparação pareada, curva Recall@k, análise de erros e varredura de parâmetros."""

from __future__ import annotations

import math
import random
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from .evaluate import recall_em_k
from .retrieve import MODOS, Recuperador
from .utils import ErroUsuario, anexar_jsonl, ler_jsonl, tabela_markdown, tokenizar_bm25

CATEGORIAS_ERRO = (
    "falha_recuperacao",
    "falha_leitura",
    "abstencao_indevida",
    "alucinacao",
    "erro_gold",
)


# ---------------------------------------------------------------------------
# Comparação pareada
# ---------------------------------------------------------------------------

def acerto_binario(r: dict, criterio: str) -> bool:
    if criterio == "juiz" and r.get("nota_juiz") is not None:
        return r["nota_juiz"] == 2
    return r["em"] == 1.0 if criterio == "em" else r["f1"] >= 0.5


def por_pergunta(registros: list[dict], config: str, criterio: str) -> dict[str, dict]:
    """Agrega as repetições por pergunta: F1 médio e acerto por maioria."""
    grupos: dict[str, list[dict]] = defaultdict(list)
    for r in registros:
        if r["config"] == config:
            grupos[r["q_id"]].append(r)
    return {
        q: {
            "f1": statistics.mean(x["f1"] for x in rs),
            "acerto": sum(acerto_binario(x, criterio) for x in rs) * 2 > len(rs),
        }
        for q, rs in grupos.items()
    }


def bootstrap_diferenca(a: list[float], b: list[float], n: int = 10000, seed: int = 42, alfa: float = 0.05) -> dict:
    """IC por bootstrap pareado para média(b - a)."""
    difs = [y - x for x, y in zip(a, b)]
    if not difs:
        return {"diferenca": None, "ic_inf": None, "ic_sup": None}
    rng = random.Random(seed)
    m = len(difs)
    medias = sorted(sum(difs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {
        "diferenca": round(statistics.mean(difs), 4),
        "ic_inf": round(medias[int(alfa / 2 * n)], 4),
        "ic_sup": round(medias[int((1 - alfa / 2) * n) - 1], 4),
    }


def mcnemar_exato(b: int, c: int) -> float:
    """Teste de McNemar exato (binomial bicaudal) sobre os pares discordantes b e c."""
    n = b + c
    if n == 0:
        return 1.0
    cauda = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return round(min(1.0, 2 * cauda), 6)


def comparar(registros: list[dict], cfg_a: str, cfg_b: str, criterio: str = "juiz", seed: int = 42) -> dict:
    pa, pb = por_pergunta(registros, cfg_a, criterio), por_pergunta(registros, cfg_b, criterio)
    qs = sorted(set(pa) & set(pb))
    boot = bootstrap_diferenca([pa[q]["f1"] for q in qs], [pb[q]["f1"] for q in qs], seed=seed)
    b = sum(pa[q]["acerto"] and not pb[q]["acerto"] for q in qs)   # só A acerta
    c = sum(pb[q]["acerto"] and not pa[q]["acerto"] for q in qs)   # só B acerta
    return {
        "a": cfg_a, "b": cfg_b, "n": len(qs), "criterio_acerto": criterio,
        "f1_a": round(statistics.mean(pa[q]["f1"] for q in qs), 4) if qs else None,
        "f1_b": round(statistics.mean(pb[q]["f1"] for q in qs), 4) if qs else None,
        "dif_f1_b_menos_a": boot["diferenca"], "ic95_inf": boot["ic_inf"], "ic95_sup": boot["ic_sup"],
        "acerto_a": round(sum(pa[q]["acerto"] for q in qs) / len(qs), 4) if qs else None,
        "acerto_b": round(sum(pb[q]["acerto"] for q in qs) / len(qs), 4) if qs else None,
        "so_a_acerta": b, "so_b_acerta": c, "mcnemar_p": mcnemar_exato(b, c),
    }


def tabela_comparacoes(comps: list[dict]) -> str:
    linhas = [
        [f"{c['a']} → {c['b']}", c["n"], c["f1_a"], c["f1_b"], c["dif_f1_b_menos_a"],
         f"[{c['ic95_inf']:.3f}, {c['ic95_sup']:.3f}]" if c["ic95_inf"] is not None else "",
         c["so_a_acerta"], c["so_b_acerta"], c["mcnemar_p"]]
        for c in comps
    ]
    return tabela_markdown(
        ["Par", "n", "F1 A", "F1 B", "Δ F1", "IC 95% Δ", "Só A acerta", "Só B acerta", "McNemar p"], linhas
    )


# ---------------------------------------------------------------------------
# Curva Recall@k por modo de recuperação
# ---------------------------------------------------------------------------

def curva_recall(gold: list[dict], recuperador: Recuperador, ks: list[int]) -> dict[str, dict[int, float]]:
    itens = [g for g in gold if g.get("evidencia")]
    if not itens:
        raise ErroUsuario("O gold não tem itens com evidência para calcular Recall@k.")
    kmax = max(ks)
    curva: dict[str, dict[int, float]] = {}
    for modo in MODOS:
        listas = [[pid for pid, _ in recuperador.recuperar(g["pergunta"], modo, kmax)] for g in itens]
        curva[modo] = {
            k: round(statistics.mean(recall_em_k(l, g["evidencia"], k) for l, g in zip(listas, itens)), 4) for k in ks
        }
    return curva


def grafico_recall(curva: dict[str, dict[int, float]], destino: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib não instalado; gráfico não gerado (a tabela CSV foi gravada).", file=sys.stderr)
        return None
    rotulos = {"bm25": "BM25 (S1)", "denso": "Denso (S2)", "hibrido": "Híbrido RRF (S3)"}
    cores = {"bm25": "#2a78b8", "denso": "#d8762b", "hibrido": "#3a9a5b"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for modo, pontos in curva.items():
        ks = sorted(pontos)
        ax.plot(ks, [pontos[k] for k in ks], marker="o", ms=3.5, lw=2, color=cores.get(modo), label=rotulos.get(modo, modo))
    ax.set_xticks(sorted({k for pontos in curva.values() for k in pontos}))
    ax.set_xlabel("k (passagens recuperadas)")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# Análise de erros
# ---------------------------------------------------------------------------

def eh_erro(r: dict) -> bool:
    if r["tipo"] == "sem_resposta":
        return not r["absteve"]
    if r.get("nota_juiz") is not None:
        return r["nota_juiz"] < 2
    return r["f1"] < 0.5


def sugerir_categoria(r: dict) -> str:
    """Categoria sugerida automaticamente; o revisor confirma ou corrige."""
    if r["tipo"] == "sem_resposta":
        return "alucinacao"
    recuperou = (r.get("recall") or 0) > 0
    if r.get("modo") == "nenhum":
        return "abstencao_indevida" if r["absteve"] else "alucinacao"
    if not recuperou:
        return "falha_recuperacao"
    if r["absteve"]:
        return "abstencao_indevida"
    if r.get("fidelidade") == 0:
        return "alucinacao"
    return "falha_leitura"


def amostrar_erros(registros: list[dict], config: str, n: int = 50, seed: int = 42) -> list[dict]:
    candidatos = [r for r in registros if r["config"] == config and r.get("rep", 0) == 0 and eh_erro(r)]
    rng = random.Random(seed)
    amostra = rng.sample(candidatos, min(n, len(candidatos)))
    return [
        {
            "q_id": r["q_id"], "config": r["config"], "tipo": r["tipo"], "pergunta": r["pergunta"],
            "resposta_ref": r["resposta_ref"], "resposta": r["resposta"], "absteve": r["absteve"],
            "evidencia": r["evidencia"], "recuperadas": [x["id"] for x in r.get("recuperadas", [])],
            "recall": r.get("recall"), "nota_juiz": r.get("nota_juiz"), "fidelidade": r.get("fidelidade"),
            "categoria_sugerida": sugerir_categoria(r), "categoria": None, "obs": "",
        }
        for r in sorted(amostra, key=lambda x: x["q_id"])
    ]


def classificar_interativo(amostra: list[dict], arq_saida: Path, passagens: dict[str, dict]) -> None:
    if not sys.stdin.isatty():
        raise ErroUsuario("A classificação é interativa e precisa de um terminal.")
    feitos = {e["q_id"] for e in ler_jsonl(arq_saida)} if arq_saida.exists() else set()
    fila = [e for e in amostra if e["q_id"] not in feitos]
    print(f"{len(fila)} erros a classificar ({len(feitos)} já classificados). Enter aceita a sugestão; 's' sai.")
    menu = "  ".join(f"[{i}] {c}" for i, c in enumerate(CATEGORIAS_ERRO, 1))
    for n, e in enumerate(fila, 1):
        print("=" * 100)
        print(f"{e['q_id']} ({n}/{len(fila)}) tipo {e['tipo']} | recall {e['recall']} | juiz {e['nota_juiz']} | fidelidade {e['fidelidade']}")
        print(f"  PERGUNTA:   {e['pergunta']}\n  REFERÊNCIA: {e['resposta_ref']}\n  SISTEMA:    {e['resposta']}")
        print(f"  Evidência: {e['evidencia']}  |  Recuperadas: {e['recuperadas']}")
        for pid in e["evidencia"][:1]:
            if pid in passagens:
                print(textwrap.indent(textwrap.fill(passagens[pid]["texto"][:900], 100), "    "))
        print(f"  {menu}\n  Sugestão: {e['categoria_sugerida']}")
        op = input("  > ").strip().lower()
        if op == "s":
            break
        if op.isdigit() and 1 <= int(op) <= len(CATEGORIAS_ERRO):
            e["categoria"] = CATEGORIAS_ERRO[int(op) - 1]
        else:
            e["categoria"] = e["categoria_sugerida"]
        e["obs"] = input("  Observação (opcional): ").strip()
        anexar_jsonl(arq_saida, e)
    print(f"Classificações salvas em {arq_saida}")


def tabela_categorias(classificados: list[dict]) -> str:
    cont = Counter(e.get("categoria") or e["categoria_sugerida"] for e in classificados)
    total = sum(cont.values()) or 1
    linhas = [[c, cont.get(c, 0), round(cont.get(c, 0) / total, 3)] for c in CATEGORIAS_ERRO]
    manuais = [e for e in classificados if e.get("categoria")]
    rodape = ""
    if manuais:
        concord = sum(e["categoria"] == e["categoria_sugerida"] for e in manuais)
        rodape = f"\n\nConcordância entre categoria sugerida e manual: {concord}/{len(manuais)}"
    return tabela_markdown(["Categoria", "n", "fração"], linhas) + rodape


# ---------------------------------------------------------------------------
# Varredura: mapeamento de evidências entre índices com segmentações diferentes
# ---------------------------------------------------------------------------

def mapear_evidencias(
    gold: list[dict], base: dict[str, dict], nova: list[dict], limiar: float = 0.5
) -> dict[str, list[str]]:
    """Para cada item, as passagens do novo índice que cobrem a evidência original.

    Uma passagem nova é relevante se, na mesma seção, o coeficiente de sobreposição de tokens
    |A∩B| / min(|A|, |B|) com alguma passagem-evidência original for >= limiar.
    """
    por_secao: dict[str, list[tuple[str, set]]] = defaultdict(list)
    for p in nova:
        por_secao[p["secao"]].append((p["id"], set(tokenizar_bm25(p["texto"]))))
    mapa: dict[str, list[str]] = {}
    for g in gold:
        ids: list[str] = []
        for e in g.get("evidencia", []):
            if e not in base:
                continue
            alvo = set(tokenizar_bm25(base[e]["texto"]))
            for pid, toks in por_secao.get(base[e]["secao"], []):
                menor = min(len(alvo), len(toks)) or 1
                if len(alvo & toks) / menor >= limiar and pid not in ids:
                    ids.append(pid)
        mapa[g["id"]] = ids
    return mapa
