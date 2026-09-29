"""Etapa 5: geração de perguntas candidatas, revisão manual, concordância e divisão dev/teste."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from .evaluate import f1_tokens, kappa_cohen
from .ollama_client import OllamaClient
from .utils import (
    FRASE_ABSTENCAO,
    ErroUsuario,
    anexar_jsonl,
    carregar_prompt,
    ler_jsonl,
    renderizar,
)

FRACAO_SEM_RESPOSTA = 0.2
MIN_PALAVRAS_PASSAGEM = 40
DESCRICAO_TIPO = {
    "factual": "um fato, valor, prazo, limite, nome ou especificação presente no trecho",
    "procedimental": "como realizar uma tarefa, a ordem de passos ou a ação a tomar em uma situação",
}
DECISOES = ("aprovar", "corrigir", "descartar")


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------

def amostrar_estratificado(passagens: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Rodízio entre seções (ordem embaralhada), sorteando passagens sem reposição dentro de cada seção."""
    rng = random.Random(seed)
    grupos: dict[str, list[dict]] = defaultdict(list)
    for p in passagens:
        if p["n_palavras"] >= MIN_PALAVRAS_PASSAGEM:
            grupos[p["secao"]].append(p)
    secoes = list(grupos)
    rng.shuffle(secoes)
    for s in secoes:
        rng.shuffle(grupos[s])
    escolhidas: list[dict] = []
    while len(escolhidas) < n and any(grupos.values()):
        for s in secoes:
            if grupos[s] and len(escolhidas) < n:
                escolhidas.append(grupos[s].pop())
    return escolhidas


def _json_da_resposta(texto: str) -> dict | None:
    try:
        dados = json.loads(texto)
        return dados if isinstance(dados, dict) else None
    except json.JSONDecodeError:
        ini, fim = texto.find("{"), texto.rfind("}")
        if ini >= 0 and fim > ini:
            try:
                return json.loads(texto[ini : fim + 1])
            except json.JSONDecodeError:
                return None
    return None


def _gerar_json(cliente: OllamaClient, modelo: str, prompt: str, seed: int, num_predict: int) -> dict | None:
    for temperatura in (0.0, 0.4):
        saida = cliente.chat(modelo, prompt, temperature=temperatura, seed=seed, num_predict=num_predict, formato="json")
        dados = _json_da_resposta(saida.texto)
        if dados:
            return dados
    return None


def gerar_candidatas(
    passagens: list[dict],
    n: int,
    cliente: OllamaClient,
    modelo: str,
    seed: int = 42,
    recuperador_bm25=None,
) -> list[dict]:
    n_sem = round(n * FRACAO_SEM_RESPOSTA)
    n_com = n - n_sem
    tpl = carregar_prompt("gerador_perguntas.txt")
    candidatas: list[dict] = []

    amostra = amostrar_estratificado(passagens, n_com, seed)
    if len(amostra) < n_com:
        print(f"Aviso: apenas {len(amostra)} passagens elegíveis para {n_com} perguntas com resposta.", file=sys.stderr)
    for i, p in enumerate(amostra):
        tipo = "procedimental" if i % 5 in (1, 3) else "factual"
        prompt = renderizar(
            tpl, tipo=tipo, descricao_tipo=DESCRICAO_TIPO[tipo], secao=p["secao"], pagina=p["pagina"], texto=p["texto"]
        )
        dados = _gerar_json(cliente, modelo, prompt, seed + i, 200)
        print(f"\r  perguntas com resposta: {i + 1}/{len(amostra)}   ", end="", file=sys.stderr, flush=True)
        if not dados or not str(dados.get("pergunta", "")).strip():
            continue
        candidatas.append({
            "pergunta": str(dados["pergunta"]).strip(),
            "resposta_ref": str(dados.get("resposta", "")).strip(),
            "evidencia": [p["id"]],
            "tipo": tipo,
            "secao": p["secao"],
            "pagina": p["pagina"],
        })
    print(file=sys.stderr)

    tpl_sem = carregar_prompt("gerador_sem_resposta.txt")
    titulos = list(dict.fromkeys(p["secao"] for p in passagens))[:80]
    rng = random.Random(seed)
    vistas = {c["pergunta"].lower() for c in candidatas}
    sem_resposta: list[str] = []
    tentativas = 0
    while len(sem_resposta) < n_sem and tentativas < max(5, n_sem):
        tentativas += 1
        exemplo = " ".join(rng.choice(passagens)["texto"].split()[:80])
        lote = min(10, n_sem - len(sem_resposta))
        prompt = renderizar(tpl_sem, n=lote, titulos="\n".join(f"- {t}" for t in titulos), exemplo=exemplo)
        dados = _gerar_json(cliente, modelo, prompt, seed + 1000 + tentativas, 600) or {}
        for q in dados.get("perguntas", []):
            q = str(q).strip()
            if q and q.lower() not in vistas and len(sem_resposta) < n_sem:
                vistas.add(q.lower())
                sem_resposta.append(q)
        print(f"\r  perguntas sem resposta: {len(sem_resposta)}/{n_sem}   ", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    for q in sem_resposta:
        item = {
            "pergunta": q,
            "resposta_ref": FRASE_ABSTENCAO,
            "evidencia": [],
            "tipo": "sem_resposta",
            "secao": None,
            "pagina": None,
        }
        if recuperador_bm25 is not None:
            # Passagens mais parecidas, para o revisor confirmar que a resposta de fato não está no manual.
            item["verificar_passagens"] = [pid for pid, _ in recuperador_bm25.bm25(q, 3)]
        candidatas.append(item)

    for i, c in enumerate(candidatas, 1):
        c["id"] = f"c{i:04d}"
        c["gerado_por"] = modelo
        c["status"] = "pendente"
    return [{"id": c.pop("id"), **c} for c in candidatas]


# ---------------------------------------------------------------------------
# Revisão manual (interativa)
# ---------------------------------------------------------------------------

def subconjunto_comum(ids: list[str], fracao: float = 0.3) -> set[str]:
    """Subconjunto determinístico (por hash do id) anotado pelos dois revisores."""
    ordenados = sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())
    return set(ordenados[: max(1, round(len(ids) * fracao))])


def _mostrar(texto: str, largura: int = 100, limite: int = 1800) -> str:
    texto = texto if len(texto) <= limite else texto[:limite] + " […]"
    return "\n".join(textwrap.fill(par, largura, initial_indent="    ", subsequent_indent="    ")
                     for par in texto.split("\n"))


def _editar(rotulo: str, atual: str) -> str:
    novo = input(f"  {rotulo} [{atual}]: ").strip()
    return novo or atual


def revisar(candidatas: list[dict], passagens: dict[str, dict], arq_revisao: Path, anotador: str, somente: set[str] | None) -> None:
    if not sys.stdin.isatty():
        raise ErroUsuario("A revisão é interativa e precisa de um terminal.")
    feitas = {r["id"] for r in ler_jsonl(arq_revisao)} if arq_revisao.exists() else set()
    fila = [c for c in candidatas if c["id"] not in feitas and (somente is None or c["id"] in somente)]
    total = len(fila) + len([c for c in candidatas if c["id"] in feitas and (somente is None or c["id"] in somente)])
    print(f"\n{len(fila)} candidatas pendentes para {anotador} ({total - len(fila)}/{total} já revisadas).")
    print("Comandos: [a]provar  [e]ditar/corrigir  [d]escartar  [p]ular  [s]air\n")

    for n, c in enumerate(fila, 1):
        print("=" * 100)
        print(f"{c['id']}  ({n}/{len(fila)})  tipo: {c['tipo']}")
        for pid in c.get("evidencia", []):
            p = passagens.get(pid)
            if p:
                print(f"  Evidência {pid} — seção {p['secao']}, p. {p['pagina']}:\n{_mostrar(p['texto'])}")
        for pid in c.get("verificar_passagens", []):
            p = passagens.get(pid)
            if p:
                print(f"  Passagem parecida {pid} — seção {p['secao']}, p. {p['pagina']}:\n{_mostrar(p['texto'], limite=600)}")
        print(f"\n  PERGUNTA: {c['pergunta']}\n  RESPOSTA: {c['resposta_ref']}\n")

        while True:
            op = input("  > ").strip().lower()[:1]
            if op in {"a", "e", "d", "p", "s"}:
                break
            print("  Opção inválida.")
        if op == "s":
            break
        if op == "p":
            continue
        registro = {
            "id": c["id"], "anotador": anotador, "pergunta": c["pergunta"], "resposta_ref": c["resposta_ref"],
            "tipo": c["tipo"], "evidencia": list(c.get("evidencia", [])), "obs": "",
        }
        if op == "a":
            registro["decisao"] = "aprovar"
        elif op == "d":
            registro["decisao"] = "descartar"
            registro["obs"] = input("  Motivo (trivial, ambígua, errada...): ").strip()
        else:
            registro["decisao"] = "corrigir"
            registro["pergunta"] = _editar("Pergunta", c["pergunta"])
            registro["resposta_ref"] = _editar("Resposta", c["resposta_ref"])
            while True:
                tipo = _editar("Tipo (factual/procedimental/sem_resposta)", c["tipo"])
                if tipo in {"factual", "procedimental", "sem_resposta"}:
                    break
            registro["tipo"] = tipo
            if tipo == "sem_resposta":
                registro["resposta_ref"], registro["evidencia"] = FRASE_ABSTENCAO, []
            else:
                ev = _editar("Evidência (ids separados por espaço)", " ".join(registro["evidencia"]))
                registro["evidencia"] = [e for e in ev.split() if e]
                invalidas = [e for e in registro["evidencia"] if e not in passagens]
                if invalidas:
                    print(f"  Aviso: ids fora do índice atual: {invalidas}")
            registro["obs"] = input("  Observação (opcional): ").strip()
        anexar_jsonl(arq_revisao, registro)
    print(f"\nRevisões salvas em {arq_revisao}")


# ---------------------------------------------------------------------------
# Concordância entre anotadores
# ---------------------------------------------------------------------------

def concordancia_anotadores(rev_a: list[dict], rev_b: list[dict]) -> dict:
    a = {r["id"]: r for r in rev_a}
    b = {r["id"]: r for r in rev_b}
    comuns = sorted(set(a) & set(b))
    if not comuns:
        raise ErroUsuario("Os dois arquivos de revisão não têm itens em comum.")
    manter_a = [a[i]["decisao"] != "descartar" for i in comuns]
    manter_b = [b[i]["decisao"] != "descartar" for i in comuns]
    mantidos = [i for i in comuns if a[i]["decisao"] != "descartar" and b[i]["decisao"] != "descartar"]
    tipos_a = [a[i]["tipo"] for i in mantidos]
    tipos_b = [b[i]["tipo"] for i in mantidos]
    return {
        "anotadores": [rev_a[0].get("anotador"), rev_b[0].get("anotador")],
        "itens_comuns": len(comuns),
        "decisao_manter": {
            "concordancia": round(sum(x == y for x, y in zip(manter_a, manter_b)) / len(comuns), 4),
            "kappa": kappa_cohen(manter_a, manter_b),
        },
        "tipo": {
            "itens": len(mantidos),
            "concordancia": round(sum(x == y for x, y in zip(tipos_a, tipos_b)) / len(mantidos), 4) if mantidos else None,
            "kappa": kappa_cohen(tipos_a, tipos_b) if mantidos else None,
        },
        "resposta_f1_medio": round(
            sum(f1_tokens(a[i]["resposta_ref"], b[i]["resposta_ref"]) for i in mantidos) / len(mantidos), 4
        ) if mantidos else None,
        "evidencia_igual": round(
            sum(set(a[i]["evidencia"]) == set(b[i]["evidencia"]) for i in mantidos) / len(mantidos), 4
        ) if mantidos else None,
    }


# ---------------------------------------------------------------------------
# Consolidação e divisão dev/teste por seção
# ---------------------------------------------------------------------------

def consolidar(candidatas: list[dict], revisoes: list[list[dict]]) -> tuple[list[dict], dict]:
    """Aplica as revisões (a primeira lista tem prioridade) e devolve os itens mantidos."""
    decisao: dict[str, dict] = {}
    for rev in reversed(revisoes):
        for r in rev:
            decisao[r["id"]] = r
    mantidos, contagem = [], Counter()
    for c in candidatas:
        r = decisao.get(c["id"])
        if r is None:
            contagem["nao_revisadas"] += 1
            continue
        contagem[r["decisao"]] += 1
        if r["decisao"] == "descartar":
            continue
        mantidos.append({
            "origem": c["id"],
            "pergunta": r["pergunta"],
            "resposta_ref": r["resposta_ref"],
            "evidencia": r["evidencia"],
            "tipo": r["tipo"],
        })
    return mantidos, dict(contagem)


def dividir_por_secao(
    itens: list[dict], secao_de: dict[str, str], frac_dev: float = 0.3, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Divide dev/teste por seção do manual: seções ligadas por uma mesma pergunta ficam no mesmo lado."""
    pai: dict[str, str] = {}

    def achar(x: str) -> str:
        pai.setdefault(x, x)
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    for it in itens:
        secoes = [secao_de.get(e, f"?{e}") for e in it["evidencia"]]
        for s in secoes:
            achar(s)
        for s in secoes[1:]:
            pai[achar(s)] = achar(secoes[0])

    grupos: dict[str, list[dict]] = defaultdict(list)
    sem_secao: list[dict] = []
    for it in itens:
        if it["evidencia"]:
            grupos[achar(secao_de.get(it["evidencia"][0], f"?{it['evidencia'][0]}"))].append(it)
        else:
            sem_secao.append(it)

    rng = random.Random(seed)
    chaves = sorted(grupos)
    rng.shuffle(chaves)
    alvo = frac_dev * sum(len(g) for g in grupos.values())
    dev, teste = [], []
    for ch in chaves:
        (dev if len(dev) + len(grupos[ch]) / 2 <= alvo else teste).extend(grupos[ch])

    rng.shuffle(sem_secao)
    n_dev = round(frac_dev * len(sem_secao))
    dev += sem_secao[:n_dev]
    teste += sem_secao[n_dev:]
    return dev, teste


def montar_gold(itens: list[dict], dev: list[dict], teste: list[dict]) -> tuple[list[dict], list[dict]]:
    ordem = {id(it): n for n, it in enumerate(itens, 1)}

    def finalizar(lista: list[dict], split: str) -> list[dict]:
        saida = []
        for it in sorted(lista, key=lambda x: ordem[id(x)]):
            saida.append({
                "id": f"q{ordem[id(it)]:03d}",
                "pergunta": it["pergunta"],
                "resposta_ref": it["resposta_ref"],
                "evidencia": it["evidencia"],
                "tipo": it["tipo"],
                "split": split,
                "origem": it["origem"],
            })
        return saida

    return finalizar(dev, "dev"), finalizar(teste, "test")


def resumo_divisao(dev: list[dict], teste: list[dict], secao_de: dict[str, str]) -> dict:
    def secoes(lista):
        return {secao_de.get(e) for it in lista for e in it["evidencia"]}

    return {
        "dev": {"itens": len(dev), "tipos": dict(Counter(i["tipo"] for i in dev)), "secoes": len(secoes(dev))},
        "test": {"itens": len(teste), "tipos": dict(Counter(i["tipo"] for i in teste)), "secoes": len(secoes(teste))},
        "secoes_em_ambos": sorted(s for s in secoes(dev) & secoes(teste) if s),
    }

