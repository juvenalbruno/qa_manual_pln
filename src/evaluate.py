"""Etapa 6: métricas de recuperação e resposta, juiz local, abstenção, latência e orquestração."""

from __future__ import annotations

import hashlib
import math
import platform
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .generate import aquecer, formatar_passagens, responder
from .index import Indice, verificar_manifesto
from .ollama_client import OllamaClient
from .retrieve import Recuperador
from .utils import (
    ErroUsuario,
    anexar_jsonl,
    carregar_prompt,
    escrever_json,
    escrever_jsonl,
    ler_json,
    ler_jsonl,
    normalizar_resposta,
    renderizar,
    tabela_markdown,
)

TIPOS = ("factual", "procedimental", "sem_resposta")


# ---------------------------------------------------------------------------
# Métricas por item
# ---------------------------------------------------------------------------

def exact_match(pred: str, ref: str) -> float:
    return float(normalizar_resposta(pred) == normalizar_resposta(ref))


def f1_tokens(pred: str, ref: str) -> float:
    p, r = normalizar_resposta(pred).split(), normalizar_resposta(ref).split()
    if not p or not r:
        return float(p == r)
    comuns = sum((Counter(p) & Counter(r)).values())
    if comuns == 0:
        return 0.0
    precisao, revocacao = comuns / len(p), comuns / len(r)
    return 2 * precisao * revocacao / (precisao + revocacao)


def recall_em_k(recuperadas: list[str], evidencia: list[str], k: int) -> float | None:
    if not evidencia:
        return None
    return len(set(recuperadas[:k]) & set(evidencia)) / len(set(evidencia))


def rr(recuperadas: list[str], evidencia: list[str]) -> float | None:
    if not evidencia:
        return None
    for pos, pid in enumerate(recuperadas, 1):
        if pid in evidencia:
            return 1.0 / pos
    return 0.0


def _nota(texto: str, validos: tuple[int, ...]) -> int | None:
    m = re.search(r"\b([0-9])\b", texto)
    if m and int(m.group(1)) in validos:
        return int(m.group(1))
    return None


def pontuar(item: dict, reg: dict, por_id: dict | None = None) -> dict:
    """Métricas automáticas de um item (sem juiz)."""
    sem_resposta = item["tipo"] == "sem_resposta"
    ids = [r["id"] for r in reg.get("recuperadas", [])]
    evid = item.get("evidencia", [])
    k = reg.get("k") or len(ids)
    com_recuperacao = reg.get("modo") != "nenhum"
    pontos: dict = {
        "tipo": item["tipo"],
        "resposta_ref": item["resposta_ref"],
        "evidencia": evid,
        "recall": recall_em_k(ids, evid, k) if com_recuperacao else None,
        "rr": rr(ids[:k], evid) if com_recuperacao else None,
    }
    if sem_resposta:
        pontos["em"] = pontos["f1"] = float(reg["absteve"])
    elif reg["absteve"]:
        pontos["em"] = pontos["f1"] = 0.0
    else:
        pred = reg.get("resposta_curta") or reg["resposta"]
        pontos["em"] = exact_match(pred, item["resposta_ref"])
        pontos["f1"] = f1_tokens(pred, item["resposta_ref"])
    # Citação correta: página citada dentro do intervalo de páginas de alguma passagem-evidência.
    pontos["citacao_correta"] = None
    if not sem_resposta and not reg["absteve"] and por_id and evid and com_recuperacao:
        pag = reg.get("pagina_citada")
        faixas = [(por_id[e]["pagina"], por_id[e].get("pagina_fim", por_id[e]["pagina"])) for e in evid if e in por_id]
        pontos["citacao_correta"] = float(pag is not None and any(a <= pag <= b for a, b in faixas))
    return pontos


# ---------------------------------------------------------------------------
# Juiz local
# ---------------------------------------------------------------------------

class Juiz:
    def __init__(self, cliente: OllamaClient, modelo: str, seed: int = 42, num_ctx: int = 8192):
        self.cliente = cliente
        self.modelo = modelo
        self.seed = seed
        self.num_ctx = num_ctx
        self.cache: dict[str, int | None] = {}
        self.tpl_correcao = carregar_prompt("juiz.txt")
        self.tpl_fidelidade = carregar_prompt("juiz_fidelidade.txt")

    def _perguntar(self, prompt: str, validos: tuple[int, ...]) -> int | None:
        chave = hashlib.sha256(prompt.encode()).hexdigest()
        if chave not in self.cache:
            saida = self.cliente.chat(
                self.modelo, prompt, temperature=0, seed=self.seed, num_ctx=self.num_ctx, num_predict=5
            )
            self.cache[chave] = _nota(saida.texto, validos)
        return self.cache[chave]

    def correcao(self, pergunta: str, resposta_ref: str, resposta: str) -> int | None:
        prompt = renderizar(self.tpl_correcao, pergunta=pergunta, resposta_ref=resposta_ref, resposta=resposta)
        return self._perguntar(prompt, (0, 1, 2))

    def fidelidade(self, resposta: str, passagens: list[dict]) -> int | None:
        prompt = renderizar(self.tpl_fidelidade, resposta=resposta, passagens=formatar_passagens(passagens))
        return self._perguntar(prompt, (0, 1))


def julgar_registro(juiz: Juiz, reg: dict, indice: Indice | None) -> dict:
    """Nota do juiz (0/1/2) e fidelidade (0/1). Casos determinísticos não chamam o modelo."""
    sem_resposta = reg["tipo"] == "sem_resposta"
    if sem_resposta:
        nota = 2 if reg["absteve"] else 0
    elif reg["absteve"]:
        nota = 0
    else:
        nota = juiz.correcao(reg["pergunta"], reg["resposta_ref"], reg.get("resposta_curta") or reg["resposta"])
    fidelidade = None
    if not reg["absteve"] and reg.get("recuperadas") and indice is not None:
        passagens = [indice.por_id[r["id"]] for r in reg["recuperadas"] if r["id"] in indice.por_id]
        fidelidade = juiz.fidelidade(reg["resposta"], passagens)
    return {"nota_juiz": nota, "fidelidade": fidelidade}


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------

def _media(valores: list) -> float | None:
    v = [x for x in valores if x is not None]
    return round(float(statistics.mean(v)), 4) if v else None


def percentil(valores: list[float], q: float) -> float | None:
    if not valores:
        return None
    v = sorted(valores)
    pos = (len(v) - 1) * q
    a, b = math.floor(pos), math.ceil(pos)
    return round(v[a] + (v[b] - v[a]) * (pos - a), 3)


def metricas(registros: list[dict]) -> dict:
    """Métricas agregadas de um conjunto de registros (uma configuração, uma repetição)."""
    if not registros:
        return {}
    k = max((r.get("k") or 0) for r in registros)
    respondiveis = [r for r in registros if r["tipo"] != "sem_resposta"]
    sem_resp = [r for r in registros if r["tipo"] == "sem_resposta"]
    lat = [r["latencia_s"] for r in registros]
    m = {
        f"recall@{k}" if k else "recall@k": _media([r["recall"] for r in respondiveis]) if k else None,
        "mrr": _media([r["rr"] for r in respondiveis]) if k else None,
        "em": _media([r["em"] for r in registros]),
        "f1": _media([r["f1"] for r in registros]),
        "em_respondiveis": _media([r["em"] for r in respondiveis]),
        "f1_respondiveis": _media([r["f1"] for r in respondiveis]),
        "juiz_media": _media([r.get("nota_juiz") for r in registros]),
        "juiz_acerto": _media([None if r.get("nota_juiz") is None else float(r["nota_juiz"] == 2) for r in registros]),
        "fidelidade": _media([r.get("fidelidade") for r in registros]),
        "citacao_correta": _media([r.get("citacao_correta") for r in respondiveis]),
        "abst_correta": _media([float(r["absteve"]) for r in sem_resp]),
        "abst_indevida": _media([float(r["absteve"]) for r in respondiveis]),
        "latencia_mediana_s": percentil(lat, 0.5),
        "latencia_p95_s": percentil(lat, 0.95),
        "n_itens": len(registros),
    }
    for tipo in TIPOS:
        sub = [r for r in registros if r["tipo"] == tipo]
        if sub:
            m[f"f1_{tipo}"] = _media([r["f1"] for r in sub])
    return m


def agregar_repeticoes(por_rep: list[dict]) -> dict:
    """Média entre repetições, com desvio-padrão amostral em 'desvio'."""
    if len(por_rep) == 1:
        return {**por_rep[0], "desvio": {}, "repeticoes": 1}
    saida: dict = {}
    desvio: dict = {}
    for chave in por_rep[0]:
        valores = [m.get(chave) for m in por_rep]
        if all(isinstance(v, (int, float)) for v in valores):
            saida[chave] = round(statistics.mean(valores), 4)
            desvio[chave] = round(statistics.stdev(valores), 4)
        else:
            saida[chave] = por_rep[0].get(chave)
    saida["desvio"] = desvio
    saida["repeticoes"] = len(por_rep)
    return saida


def reprodutibilidade(registros: list[dict]) -> float | None:
    """Fração das perguntas cujas respostas são idênticas em todas as repetições."""
    por_q: dict[str, set] = defaultdict(set)
    reps = {r["rep"] for r in registros}
    if len(reps) < 2:
        return None
    for r in registros:
        por_q[r["q_id"]].add(r["resposta"])
    return round(sum(len(s) == 1 for s in por_q.values()) / len(por_q), 4)


def calcular_metricas_run(registros: list[dict]) -> dict:
    por_cfg: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in registros:
        por_cfg[r["config"]][r.get("rep", 0)].append(r)
    saida = {}
    for cfg, reps in por_cfg.items():
        agg = agregar_repeticoes([metricas(reps[i]) for i in sorted(reps)])
        agg["reprodutibilidade_respostas"] = reprodutibilidade([r for rs in reps.values() for r in rs])
        saida[cfg] = agg
    return saida


COLUNAS_TABELA = [
    ("recall@k", "Recall@k"), ("mrr", "MRR"), ("em", "EM"), ("f1", "F1"),
    ("juiz_media", "Juiz (0-2)"), ("fidelidade", "Fidelidade"), ("abst_correta", "Abst. correta"),
    ("abst_indevida", "Abst. indevida"), ("latencia_mediana_s", "Lat. mediana (s)"), ("latencia_p95_s", "Lat. p95 (s)"),
]


def _valor_recall(m: dict):
    for chave, v in m.items():
        if chave.startswith("recall@"):
            return chave, v
    return None, None


def tabela_metricas(metrics: dict) -> str:
    linhas = []
    for cfg in sorted(c for c in metrics if not c.startswith("_")):
        m = metrics[cfg]
        linha = [cfg]
        for chave, _ in COLUNAS_TABELA:
            if chave == "recall@k":
                chave, v = _valor_recall(m)
            else:
                v = m.get(chave)
            d = m.get("desvio", {}).get(chave) if chave else None
            if isinstance(v, float) and d:
                linha.append(f"{v:.3f} ± {d:.3f}")
            else:
                linha.append(v)
        linhas.append(linha)
    return tabela_markdown(["Config"] + [c for _, c in COLUNAS_TABELA], linhas)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def amostrar_gold(gold: list[dict], limite: int | None, seed: int = 42) -> list[dict]:
    if not limite or limite >= len(gold):
        return gold
    rng = random.Random(seed)
    return sorted(rng.sample(gold, limite), key=lambda g: g["id"])


def validar_gold(gold: list[dict], origem: str) -> None:
    obrig = {"id", "pergunta", "resposta_ref", "evidencia", "tipo"}
    for n, g in enumerate(gold, 1):
        faltando = obrig - g.keys()
        if faltando:
            raise ErroUsuario(f"{origem}, item {n}: campos ausentes {sorted(faltando)}.")
        if g["tipo"] not in TIPOS:
            raise ErroUsuario(f"{origem}, item {g['id']}: tipo inválido '{g['tipo']}'.")
    ids = [g["id"] for g in gold]
    if len(ids) != len(set(ids)):
        raise ErroUsuario(f"{origem}: ids duplicados.")


def executar_avaliacao(
    gold: list[dict],
    configs: list[dict],
    repeticoes: int,
    run_dir: Path,
    run_id: str,
    usar_juiz: bool = True,
    dir_indice: Path | None = None,
    k_override: int | None = None,
    mapa_evidencias: dict[str, list[str]] | None = None,
    meta_extra: dict | None = None,
) -> dict:
    """Roda as configurações sobre o gold, grava respostas.jsonl e metrics.json e devolve as métricas.

    Se run_dir já contém respostas.jsonl, os itens já executados são pulados (retomada).
    Fase 1 gera todas as respostas; fase 2 roda o juiz. Separar as fases evita alternar
    leitor e juiz na memória (importante com 8-16 GB de RAM) e não contamina a latência.
    """
    arq_resp = run_dir / "respostas.jsonl"
    arq_julg = run_dir / "julgamentos.jsonl"
    feitos = {(r["config"], r.get("rep", 0), r["q_id"]) for r in ler_jsonl(arq_resp)} if arq_resp.exists() else set()

    precisa_indice = any(c["modo"] != "nenhum" for c in configs)
    dir_indice = Path(dir_indice or configs[0].get("indice", "index"))
    indice = None
    if precisa_indice:
        for c in configs:
            if c["modo"] != "nenhum":
                verificar_manifesto(dir_indice, c.get("embeddings"))
        indice = Indice(dir_indice)

    clientes: dict[str, OllamaClient] = {}

    def cliente_de(cfg: dict) -> OllamaClient:
        url = cfg.get("ollama_url", "http://localhost:11434")
        if url not in clientes:
            clientes[url] = OllamaClient(url)
        return clientes[url]

    total = len(gold) * len(configs) * repeticoes
    feito = 0
    for cfg in configs:
        cli = cliente_de(cfg)
        rec = Recuperador(indice, cli, cfg.get("rrf_k", 60)) if indice is not None else None
        pendentes = [(rep, g) for rep in range(repeticoes) for g in gold if (cfg["nome"], rep, g["id"]) not in feitos]
        if pendentes:
            aquecer(cli, cfg["leitor"], cfg.get("think"))
        for rep, g in pendentes:
            reg = responder(g["pergunta"], cfg, rec, cli, k=k_override)
            item = dict(g)
            if mapa_evidencias is not None:
                item["evidencia"] = mapa_evidencias.get(g["id"], [])
            reg = {
                "run_id": run_id,
                "rep": rep,
                "q_id": g["id"],
                **reg,
                **pontuar(item, reg, indice.por_id if indice else None),
            }
            anexar_jsonl(arq_resp, reg)
            feito += 1
            print(f"\r  [{cfg['nome']}] rep {rep + 1}/{repeticoes}: {feito}/{total - len(feitos)} "
                  f"(última: {reg['latencia_s']:.1f}s)   ", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    registros = ler_jsonl(arq_resp)
    chaves_gold = {g["id"] for g in gold}
    nomes = {c["nome"] for c in configs}
    registros = [r for r in registros if r["q_id"] in chaves_gold and r["config"] in nomes]

    if usar_juiz:
        cfg0 = configs[0]
        juiz = Juiz(cliente_de(cfg0), cfg0["juiz"], cfg0.get("seed", 42), cfg0.get("num_ctx", 8192))
        julgados = {(j["config"], j["rep"], j["q_id"]): j for j in ler_jsonl(arq_julg)} if arq_julg.exists() else {}
        faltam = [r for r in registros if (r["config"], r["rep"], r["q_id"]) not in julgados]
        if faltam:
            aquecer(juiz.cliente, juiz.modelo)
        for n, r in enumerate(faltam, 1):
            j = {"config": r["config"], "rep": r["rep"], "q_id": r["q_id"], "juiz": juiz.modelo,
                 **julgar_registro(juiz, r, indice)}
            anexar_jsonl(arq_julg, j)
            julgados[(r["config"], r["rep"], r["q_id"])] = j
            print(f"\r  juiz: {n}/{len(faltam)}   ", end="", file=sys.stderr, flush=True)
        if faltam:
            print(file=sys.stderr)
        for r in registros:
            j = julgados.get((r["config"], r["rep"], r["q_id"]), {})
            r["nota_juiz"], r["fidelidade"] = j.get("nota_juiz"), j.get("fidelidade")
        escrever_jsonl(arq_resp, registros)

    metrics = calcular_metricas_run(registros)
    arq_metrics = run_dir / "metrics.json"
    anterior = ler_json(arq_metrics) if arq_metrics.exists() else {}
    metrics["_meta"] = {
        "run_id": run_id,
        "n_itens_gold": len(gold),
        "configs": [c["nome"] for c in configs],
        "repeticoes": repeticoes,
        "leitor": {c["nome"]: c["leitor"] for c in configs},
        "juiz": configs[0]["juiz"] if usar_juiz else None,
        "indice": str(dir_indice) if precisa_indice else None,
        "k": k_override or {c["nome"]: c.get("k") for c in configs},
        "plataforma": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        **(meta_extra or {}),
    }
    if "_validacao_juiz" in anterior:
        metrics["_validacao_juiz"] = anterior["_validacao_juiz"]
    escrever_json(arq_metrics, metrics)
    return metrics


# ---------------------------------------------------------------------------
# Validação do juiz
# ---------------------------------------------------------------------------

def kappa_ponderado(a: list[int], b: list[int], categorias: tuple[int, ...] = (0, 1, 2)) -> float | None:
    """Kappa de Cohen com pesos quadráticos (concordância ordinal)."""
    n = len(a)
    if n == 0:
        return None
    idx = {c: i for i, c in enumerate(categorias)}
    q = len(categorias)
    obs = [[0.0] * q for _ in range(q)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ma = [sum(obs[i]) for i in range(q)]
    mb = [sum(obs[i][j] for i in range(q)) for j in range(q)]
    w = [[((i - j) ** 2) / ((q - 1) ** 2) for j in range(q)] for i in range(q)]
    num = sum(w[i][j] * obs[i][j] for i in range(q) for j in range(q))
    den = sum(w[i][j] * ma[i] * mb[j] / n for i in range(q) for j in range(q))
    return round(1 - num / den, 4) if den else None


def kappa_cohen(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def concordancia_juiz(pares: list[tuple[int, int]]) -> dict:
    manual = [m for m, _ in pares]
    auto = [a for _, a in pares]
    n = len(pares)
    return {
        "n": n,
        "concordancia_exata": round(sum(m == a for m, a in pares) / n, 4) if n else None,
        "concordancia_binaria_correto": round(sum((m == 2) == (a == 2) for m, a in pares) / n, 4) if n else None,
        "kappa_ponderado_quadratico": kappa_ponderado(manual, auto),
        "media_manual": _media(manual),
        "media_juiz": _media(auto),
    }
