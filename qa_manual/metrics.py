"""Passo 13: métricas de resposta (EM, F1, BERTScore), recuperação (Recall@k, MRR), abstenção e ganho pareado."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from .verify import remover_citacoes

ARTIGOS_PREPOSICOES = {
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "ao",
    "à",
    "aos",
    "às",
    "e",
}
_SEPARADOR_NUMERICO = re.compile(r"(?<=\d)[.,](?=\d)")
_PONTUACAO = re.compile(r"[^\w\s]|_")
NAN = float("nan")


def normalizar(t: str) -> str:
    """Minúsculas, sem pontuação, sem artigos e preposições frequentes, espaços colapsados; mantém acentos.

    Separadores de milhar e decimal entre dígitos são removidos (``1.420`` e ``1420`` ficam iguais).
    """
    t = unicodedata.normalize("NFC", t.lower())
    t = _PONTUACAO.sub(" ", _SEPARADOR_NUMERICO.sub("", t))
    return " ".join(tok for tok in t.split() if tok not in ARTIGOS_PREPOSICOES)


def exact_match(pred: str, ref: str) -> int:
    """1 se as respostas normalizadas são idênticas."""
    return int(normalizar(pred) == normalizar(ref))


def f1_tokens(pred: str, ref: str) -> float:
    """F1 sobre multiconjuntos de tokens normalizados (estilo SQuAD).

    Se uma das respostas fica vazia após a normalização, vale 1 quando as duas estão vazias e 0 caso contrário.
    """
    p, r = normalizar(pred).split(), normalizar(ref).split()
    if not p or not r:
        return float(p == r)
    comuns = sum((Counter(p) & Counter(r)).values())
    if comuns == 0:
        return 0.0
    precisao, revocacao = comuns / len(p), comuns / len(r)
    return 2 * precisao * revocacao / (precisao + revocacao)


def recall_at_k(recuperadas_ids: list[str], evidencia_ids: list[str]) -> int:
    """1 se algum trecho-evidência está entre os recuperados (a lista já é o top-k)."""
    return int(bool(set(recuperadas_ids) & set(evidencia_ids)))


def mrr_at_k(recuperadas_ids: list[str], evidencia_ids: list[str]) -> float:
    """Inverso da posição (1-based) da primeira evidência entre os recuperados; 0 se ausente."""
    evid = set(evidencia_ids)
    for pos, tid in enumerate(recuperadas_ids, start=1):
        if tid in evid:
            return 1.0 / pos
    return 0.0


def bertscore_f1(preds: list[str], refs: list[str], modelo: str, num_layers: int) -> list[float]:
    """BERTScore F1 por item em CPU (``bert_score.score``, sem reescala pela linha de base).

    O cálculo roda em um subprocesso (:mod:`qa_manual.bertscore_proc`): no macOS, ``faiss-cpu`` e ``torch``
    trazem cópias diferentes do OpenMP e não podem ser carregados no mesmo processo. O texto passa só pelo pipe
    local. O modelo (BERTimbau) é baixado do Hugging Face na primeira execução; é download, não envio de dados.
    """
    if not preds:
        return []
    entrada = json.dumps({"preds": preds, "refs": refs, "modelo": modelo, "num_layers": num_layers})
    r = subprocess.run(
        [sys.executable, "-m", "qa_manual.bertscore_proc"], input=entrada, capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        ultima = (r.stderr or "").strip().splitlines()[-1:] or [f"código {r.returncode}"]
        raise RuntimeError(f"BERTScore falhou: {ultima[0]}")
    valores = json.loads(r.stdout)
    if len(valores) != len(preds):
        raise RuntimeError("BERTScore devolveu um número de valores diferente do de pares.")
    return [float(v) for v in valores]


# ---------------------------------------------------------------------------
# Pontuação por item e agregação
# ---------------------------------------------------------------------------


def pontuar(reg: dict, item: dict) -> dict:
    """Métricas de um registro de ``respostas.jsonl`` contra o item do gold.

    Respondíveis: EM e F1 contra ``resposta_ref`` (zero se o sistema se absteve; a citação de fonte é removida
    antes da comparação); Recall/MRR sobre ``recuperadas`` (NaN sem recuperação, em S0). Itens
    ``sem_resposta`` não têm EM/F1/Recall.
    """
    respondivel = item["tipo"] != "sem_resposta"
    pred = remover_citacoes(reg["resposta"])
    out = {
        "config": reg["config"],
        "leitor": reg["leitor"],
        "q_id": reg["q_id"],
        "repeticao": reg.get("repeticao", 1),
        "tipo": item["tipo"],
        "respondivel": respondivel,
        "absteve": bool(reg["absteve"]),
        "fonte_valida": bool(reg["fonte_valida"]),
        "latencia_s": float(reg["latencia_s"]),
        "pred": pred,
        "ref": item["resposta_ref"],
        "em": NAN,
        "f1_tokens": NAN,
        "bertscore_f1": NAN,
        "recall": NAN,
        "mrr": NAN,
    }
    if not respondivel:
        return out
    if reg["absteve"]:
        out.update(em=0, f1_tokens=0.0, bertscore_f1=0.0)
    else:
        out.update(em=exact_match(pred, item["resposta_ref"]), f1_tokens=f1_tokens(pred, item["resposta_ref"]))
    if reg.get("retriever", "nenhum") != "nenhum":
        ids = [r["id"] for r in reg["recuperadas"]]
        out.update(recall=recall_at_k(ids, item["evidencia"]), mrr=mrr_at_k(ids, item["evidencia"]))
    return out


def _media(valores: list[float]) -> float:
    validos = [v for v in valores if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(validos)) if validos else NAN


def agregar(pontuados: list[dict], k: int = 5) -> list[dict]:
    """Uma linha por par (config, leitor) com as colunas de ``metrics.csv`` (contrato 5.7).

    ``perplexidade`` sai como NaN e é preenchida por quem tiver a medida.
    """
    grupos: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pontuados:
        grupos[(p["config"], p["leitor"])].append(p)
    linhas = []
    for (config, leitor), itens in grupos.items():
        resp = [p for p in itens if p["respondivel"]]
        sem = [p for p in itens if not p["respondivel"]]
        nao_abstidas = [p for p in itens if not p["absteve"]]
        abst_correta = _media([float(p["absteve"]) for p in sem])
        latencias = [p["latencia_s"] for p in itens]
        linhas.append(
            {
                "config": config,
                "leitor": leitor,
                "n": len(itens),
                "em": _media([p["em"] for p in resp]),
                "f1_tokens": _media([p["f1_tokens"] for p in resp]),
                "bertscore_f1": _media([p["bertscore_f1"] for p in resp]),
                f"recall_at_{k}": _media([p["recall"] for p in resp]),
                f"mrr_at_{k}": _media([p["mrr"] for p in resp]),
                "abst_correta": abst_correta,
                "abst_indevida": _media([float(p["absteve"]) for p in resp]),
                "alucinacao_sem_resposta": 1 - abst_correta if not math.isnan(abst_correta) else NAN,
                "fonte_valida_pct": _media([float(p["fonte_valida"]) for p in nao_abstidas]),
                "latencia_mediana_s": float(np.median(latencias)) if latencias else NAN,
                "latencia_p95_s": float(np.percentile(latencias, 95)) if latencias else NAN,
                "perplexidade": NAN,
            }
        )
    return linhas


def bootstrap_ic(
    valores: list[float], n_reamostras: int = 1000, seed: int = 42, alfa: float = 0.05
) -> tuple[float, float, float]:
    """Média e intervalo de confiança percentil por bootstrap: ``(media, inferior, superior)``."""
    v = np.asarray(valores, dtype=float)
    if v.size == 0:
        return NAN, NAN, NAN
    rng = np.random.default_rng(seed)
    medias = v[rng.integers(0, v.size, size=(n_reamostras, v.size))].mean(axis=1)
    inf, sup = np.percentile(medias, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return float(v.mean()), float(inf), float(sup)


def ganho_pareado(pontuados: list[dict], n_reamostras: int = 1000, seed: int = 42) -> list[dict]:
    """Ganho do leitor ajustado sobre o base, pergunta a pergunta, com IC 95% por bootstrap.

    Só itens respondíveis respondidos pelos dois leitores na mesma configuração entram. Com repetições, cada
    pergunta usa a média das repetições.
    """
    por_chave: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in pontuados:
        if p["respondivel"]:
            por_chave[(p["config"], p["leitor"], p["q_id"])]["em"].append(p["em"])
            por_chave[(p["config"], p["leitor"], p["q_id"])]["f1_tokens"].append(p["f1_tokens"])
    configs = list(dict.fromkeys(c for c, _, _ in por_chave))
    linhas = []
    for config in configs:
        q_base = {q for c, lt, q in por_chave if c == config and lt == "base"}
        q_aj = {q for c, lt, q in por_chave if c == config and lt == "ajustado"}
        comuns = sorted(q_base & q_aj)
        if not comuns:
            continue
        for metrica in ("em", "f1_tokens"):
            base = [float(np.mean(por_chave[(config, "base", q)][metrica])) for q in comuns]
            aj = [float(np.mean(por_chave[(config, "ajustado", q)][metrica])) for q in comuns]
            media, inf, sup = bootstrap_ic([a - b for a, b in zip(aj, base, strict=True)], n_reamostras, seed)
            linhas.append(
                {
                    "config": config,
                    "metrica": metrica,
                    "n_perguntas": len(comuns),
                    "media_base": float(np.mean(base)),
                    "media_ajustado": float(np.mean(aj)),
                    "ganho_medio": media,
                    "ic95_inf": inf,
                    "ic95_sup": sup,
                }
            )
    return linhas
