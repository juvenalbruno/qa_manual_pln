"""Passo 12: conjunto de avaliação (gold).

Fluxo: candidatas geradas pelo LLM local (``gerar_candidatas``) -> CSV para revisão humana
(``exportar_revisao``) -> importação das decisões (``importar_revisao``) -> divisão dev/teste por tema
(``dividir``). ``reservar_ppl`` separa trechos para a medida de perplexidade.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from . import prompts
from .chunking import Trecho
from .io_utils import ErroUsuario, sha256_texto
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)

TIPOS = ("factual", "procedimental", "sem_resposta")
DECISOES = ("aceitar", "editar", "descartar")
COLUNAS_REVISAO = [
    "id",
    "tipo",
    "tema_id",
    "evidencia",
    "pergunta",
    "resposta_ref",
    "texto_evidencia",
    "decisao",
    "pergunta_corrigida",
    "resposta_corrigida",
    "observacao",
]
CAMPOS_GOLD = ("id", "pergunta", "resposta_ref", "evidencia", "tipo")
MIN_TOKENS_AMOSTRA = 120
MAX_TENTATIVAS = 3
SEPARADOR_EVIDENCIA = "|"
_OBJETO_JSON = re.compile(r"\{.*\}", re.DOTALL)
_ID_TRECHO = re.compile(r"t\d+")


# ---------------------------------------------------------------------------
# Amostragem e geração
# ---------------------------------------------------------------------------


def amostrar_trechos(trechos: list[Trecho], n: int, seed: int) -> list[Trecho]:
    """Amostra estratificada por ``tema_id``.

    A cota de cada tema é proporcional ao número de trechos dele, com pelo menos um trecho por tema quando
    ``n`` comporta todos os temas (método do maior resto). Trechos com menos de 120 tokens são excluídos.

    Returns:
        Até ``n`` trechos distintos, em ordem aleatória determinística.
    """
    rng = random.Random(seed)
    grupos: dict[str, list[Trecho]] = defaultdict(list)
    for t in trechos:
        if t.n_tokens >= MIN_TOKENS_AMOSTRA:
            grupos[t.tema_id].append(t)
    total = sum(len(g) for g in grupos.values())
    n = min(n, total)
    if n <= 0:
        return []
    temas = sorted(grupos)
    base = {t: 1 if n >= len(temas) else 0 for t in temas}
    resto = n - sum(base.values())
    capacidade = sum(len(grupos[t]) - base[t] for t in temas)
    ideal = {t: resto * (len(grupos[t]) - base[t]) / capacidade if capacidade else 0.0 for t in temas}
    cota = {t: base[t] + math.floor(ideal[t]) for t in temas}
    sorteio = {t: rng.random() for t in temas}
    ordem = sorted(temas, key=lambda t: (-(ideal[t] - math.floor(ideal[t])), sorteio[t]))
    faltam = n - sum(cota.values())
    while faltam > 0:
        for t in ordem:
            if faltam and cota[t] < len(grupos[t]):
                cota[t] += 1
                faltam -= 1
    escolhidos = [tr for t in temas for tr in rng.sample(grupos[t], cota[t])]
    rng.shuffle(escolhidos)
    return escolhidos


def extrair_json(texto: str) -> dict | None:
    """Extrai o primeiro objeto JSON da saída do gerador, tolerando texto antes e depois."""
    m = _OBJETO_JSON.search(texto)
    if not m:
        return None
    try:
        dados = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return dados if isinstance(dados, dict) else None


def gerar_json(
    client: OllamaClient, modelo: str, prompt: str, chaves: tuple[str, ...], cfg, contagem: dict
) -> dict | None:
    """Chama o gerador até ``MAX_TENTATIVAS`` vezes até obter um JSON com ``chaves`` preenchidas.

    A cada tentativa a seed muda (com temperatura 0 e a mesma seed, a saída seria idêntica). ``contagem``
    acumula ``tentativas`` e ``validas``.
    """
    for tentativa in range(MAX_TENTATIVAS):
        opcoes = cfg.decodificacao.opcoes() | {"seed": cfg.gold.seed + tentativa, "num_predict": 256}
        r = client.chat(model=modelo, messages=[{"role": "user", "content": prompt}], options=opcoes, format="json")
        contagem["tentativas"] = contagem.get("tentativas", 0) + 1
        dados = extrair_json(r.get("content", ""))
        if dados and all(isinstance(dados.get(c), str) and dados[c].strip() for c in chaves):
            contagem["validas"] = contagem.get("validas", 0) + 1
            return {c: dados[c].strip() for c in chaves}
    return None


def gerar_candidatas(cfg, client: OllamaClient, trechos: list[Trecho], n: int | None = None) -> tuple[list[dict], dict]:
    """Gera perguntas candidatas com o modelo ``modelos.gerador``.

    ``n * (1 - frac_sem_resposta)`` trechos recebem uma pergunta respondível (60% factual, 40% procedimental)
    e ``n * frac_sem_resposta`` recebem uma pergunta sem resposta no trecho. Saídas inválidas são descartadas
    depois de três tentativas.

    Returns:
        ``(candidatas, estatisticas)`` com candidatas no contrato 5.3 (mais ``trecho_origem``).
    """
    n = n or cfg.gold.n_candidatas
    n_sem = round(n * cfg.gold.frac_sem_resposta)
    n_resp = n - n_sem
    respondiveis = amostrar_trechos(trechos, n_resp, cfg.gold.seed)
    sem_resposta = amostrar_trechos(trechos, n_sem, cfg.gold.seed + 1)
    if len(respondiveis) < n_resp:
        log.warning(
            "só %d trechos com >= %d tokens; menos candidatas que o pedido", len(respondiveis), MIN_TOKENS_AMOSTRA
        )

    tarefas = [("procedimental" if i % 5 in (1, 3) else "factual", t) for i, t in enumerate(respondiveis)]
    tarefas += [("sem_resposta", t) for t in sem_resposta]
    modelo = cfg.modelos.gerador
    contagem: dict = {}
    candidatas: list[dict] = []
    for tipo, t in tqdm(tarefas, desc="candidatas", unit="pergunta"):
        if tipo == "sem_resposta":
            prompt = prompts.preencher(prompts.carregar("gerador_sem_resposta"), texto=t.texto)
            dados = gerar_json(client, modelo, prompt, ("pergunta",), cfg, contagem)
        else:
            prompt = prompts.preencher(prompts.carregar("gerador_perguntas"), tipo=tipo, texto=t.texto)
            dados = gerar_json(client, modelo, prompt, ("pergunta", "resposta"), cfg, contagem)
        if dados is None:
            log.warning("gerador sem JSON válido para o trecho %s após %d tentativas", t.id, MAX_TENTATIVAS)
            continue
        candidatas.append(
            {
                "id": f"q{len(candidatas) + 1:03d}",
                "pergunta": dados["pergunta"],
                "resposta_ref": cfg.abstencao.frase if tipo == "sem_resposta" else dados["resposta"],
                "evidencia": [] if tipo == "sem_resposta" else [t.id],
                "tipo": tipo,
                "tema_id": t.tema_id,
                "origem": "gerado",
                "revisado": False,
                "trecho_origem": t.id,
            }
        )
    tentativas = contagem.get("tentativas", 0)
    stats = {
        "pedidas": n,
        "geradas": len(candidatas),
        "tentativas": tentativas,
        "json_valido_pct": round(contagem.get("validas", 0) / tentativas, 3) if tentativas else 0.0,
    }
    log.info("candidatas: %s", stats)
    return candidatas, stats


# ---------------------------------------------------------------------------
# Revisão humana (CSV)
# ---------------------------------------------------------------------------


def exportar_revisao(candidatas: list[dict], trechos: list[Trecho], destino: Path) -> int:
    """Grava o CSV de revisão (contrato 5.4) com o texto do trecho-evidência para o revisor.

    O arquivo usa ``;`` como separador e UTF-8 com BOM, para abrir direto no Excel em português e no
    LibreOffice. Em itens ``sem_resposta``, ``texto_evidencia`` mostra o trecho que originou a pergunta.

    Returns:
        Número de linhas gravadas.
    """
    por_id = {t.id: t for t in trechos}
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_REVISAO, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for c in candidatas:
            ids = c.get("evidencia") or ([c["trecho_origem"]] if c.get("trecho_origem") else [])
            textos = [por_id[i].texto for i in ids if i in por_id]
            prefixo = "[trecho de origem; a resposta NÃO deve estar no manual] " if c["tipo"] == "sem_resposta" else ""
            w.writerow(
                {
                    "id": c["id"],
                    "tipo": c["tipo"],
                    "tema_id": c.get("tema_id", ""),
                    "evidencia": SEPARADOR_EVIDENCIA.join(c.get("evidencia", [])),
                    "pergunta": c["pergunta"],
                    "resposta_ref": c["resposta_ref"],
                    "texto_evidencia": prefixo + "\n---\n".join(textos),
                    "decisao": "",
                    "pergunta_corrigida": "",
                    "resposta_corrigida": "",
                    "observacao": "",
                }
            )
    return len(candidatas)


def _ler_csv(caminho: Path) -> list[dict]:
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        amostra = f.read(4096)
        f.seek(0)
        primeira = amostra.split("\n", 1)[0]
        delimitador = max(";,\t", key=primeira.count)
        leitor = csv.DictReader(f, delimiter=delimitador)
        faltando = [c for c in COLUNAS_REVISAO if c not in (leitor.fieldnames or [])]
        if faltando:
            raise ErroUsuario(f"{caminho}: coluna(s) ausente(s): {', '.join(faltando)}.")
        return list(leitor)


def importar_revisao(csv_path: Path, frase_abstencao: str) -> list[dict]:
    """Aplica as decisões do CSV revisado.

    ``aceitar`` mantém o item; ``editar`` substitui pergunta e/ou resposta pelas colunas corrigidas;
    ``descartar`` remove. Todos os itens mantidos ficam com ``revisado: true``.

    Raises:
        ErroUsuario: se alguma linha estiver sem decisão, com decisão inválida ou inconsistente.
    """
    linhas = _ler_csv(Path(csv_path))
    sem_decisao, erros, itens, vistos = [], [], [], set()
    for n, row in enumerate(linhas, start=2):
        qid = (row.get("id") or "").strip()
        decisao = (row.get("decisao") or "").strip().lower()
        if not decisao:
            sem_decisao.append(qid or f"linha {n}")
            continue
        if decisao not in DECISOES:
            erros.append(f"linha {n} ({qid}): decisão '{decisao}' inválida; use {', '.join(DECISOES)}")
            continue
        if qid in vistos:
            erros.append(f"linha {n}: id {qid} repetido")
            continue
        vistos.add(qid)
        if decisao == "descartar":
            continue
        tipo = (row.get("tipo") or "").strip()
        if tipo not in TIPOS:
            erros.append(f"linha {n} ({qid}): tipo '{tipo}' inválido; use {', '.join(TIPOS)}")
            continue
        pergunta, resposta = row["pergunta"].strip(), row["resposta_ref"].strip()
        if decisao == "editar":
            pc, rc = row.get("pergunta_corrigida", "").strip(), row.get("resposta_corrigida", "").strip()
            if not pc and not rc:
                erros.append(f"linha {n} ({qid}): 'editar' sem pergunta_corrigida nem resposta_corrigida")
                continue
            pergunta, resposta = pc or pergunta, rc or resposta
        evidencia = _ID_TRECHO.findall(row.get("evidencia") or "")
        if tipo == "sem_resposta":
            resposta, evidencia = frase_abstencao, []
        elif not evidencia:
            erros.append(f"linha {n} ({qid}): item {tipo} sem evidência")
            continue
        itens.append(
            {
                "id": qid,
                "pergunta": pergunta,
                "resposta_ref": resposta,
                "evidencia": evidencia,
                "tipo": tipo,
                "tema_id": (row.get("tema_id") or "").strip(),
                "origem": "editado" if decisao == "editar" else "gerado",
                "revisado": True,
            }
        )
    if sem_decisao:
        exemplos = ", ".join(sem_decisao[:10]) + (" ..." if len(sem_decisao) > 10 else "")
        erros.insert(0, f"{len(sem_decisao)} linha(s) sem decisão: {exemplos}")
    if erros:
        raise ErroUsuario(f"{csv_path}: revisão incompleta ou inválida:\n  - " + "\n  - ".join(erros))
    log.info("revisão importada: %d itens mantidos de %d linhas", len(itens), len(linhas))
    return itens


# ---------------------------------------------------------------------------
# Divisão dev/teste e reserva para perplexidade
# ---------------------------------------------------------------------------


def _temas_dev(por_tema: dict[str, list[dict]], alvo: float, rng: random.Random) -> set[str]:
    temas = sorted(por_tema)
    rng.shuffle(temas)
    escolhidos, n_dev = set(), 0
    for t in temas:
        tam = len(por_tema[t])
        if abs(n_dev + tam - alvo) < abs(n_dev - alvo):
            escolhidos.add(t)
            n_dev += tam
    return escolhidos


def dividir(gold: list[dict], frac_dev: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Divide o gold em dev e teste por ``tema_id``: todos os itens de um tema caem no mesmo split.

    Os temas são embaralhados e atribuídos ao dev até ~``frac_dev`` dos itens. Se houver itens
    ``sem_resposta``, os dois splits recebem pelo menos um; quando isso é impossível respeitando os temas,
    só os ``sem_resposta`` (que não têm evidência) são redistribuídos, com aviso no log.

    Returns:
        ``(dev, test)`` com o campo ``split`` preenchido.
    """
    if not gold:
        raise ErroUsuario("gold vazio; nada a dividir.")
    por_tema: dict[str, list[dict]] = defaultdict(list)
    for it in gold:
        por_tema[it.get("tema_id", "")].append(it)
    if len(por_tema) < 2:
        raise ErroUsuario("não é possível dividir por tema: todos os itens são do mesmo tema.")
    alvo = frac_dev * len(gold)
    tem_sem = any(it["tipo"] == "sem_resposta" for it in gold)

    def ok(dev: list[dict], test: list[dict]) -> bool:
        if not dev or not test:
            return False
        return not tem_sem or all(any(it["tipo"] == "sem_resposta" for it in s) for s in (dev, test))

    for tentativa in range(200):
        temas_dev = _temas_dev(por_tema, alvo, random.Random(seed + tentativa))
        dev = [it for it in gold if it.get("tema_id", "") in temas_dev]
        test = [it for it in gold if it.get("tema_id", "") not in temas_dev]
        if ok(dev, test):
            break
    else:
        respondiveis = [it for it in gold if it["tipo"] != "sem_resposta"]
        sem = [it for it in gold if it["tipo"] == "sem_resposta"]
        por_tema_r: dict[str, list[dict]] = defaultdict(list)
        for it in respondiveis:
            por_tema_r[it.get("tema_id", "")].append(it)
        temas_dev = _temas_dev(por_tema_r, frac_dev * len(respondiveis), random.Random(seed))
        dev = [it for it in respondiveis if it.get("tema_id", "") in temas_dev]
        test = [it for it in respondiveis if it.get("tema_id", "") not in temas_dev]
        rng = random.Random(seed)
        sem = sorted(sem, key=lambda it: it["id"])
        rng.shuffle(sem)
        n_sem_dev = min(len(sem) - 1, max(1, round(frac_dev * len(sem))))
        dev += sem[:n_sem_dev]
        test += sem[n_sem_dev:]
        log.warning("itens sem_resposta redistribuídos sem respeitar o tema (não têm evidência)")
        if not dev or not test:
            raise ErroUsuario("não foi possível dividir o gold em dev e teste não vazios.")

    dev = sorted(({**it, "split": "dev"} for it in dev), key=lambda it: it["id"])
    test = sorted(({**it, "split": "test"} for it in test), key=lambda it: it["id"])
    log.info("divisão: %d dev, %d test (%d temas)", len(dev), len(test), len(por_tema))
    return dev, test


def reservar_ppl(
    trechos: list[Trecho], gold_test: list[dict] | None, n: int = 30, seed: int = 42, destino: Path | None = None
) -> list[Trecho]:
    """Escolhe ``n`` trechos de temas que não aparecem em ``gold_test`` para a medida de perplexidade.

    Args:
        trechos: todos os trechos do manual.
        gold_test: itens do teste (``None`` se ainda não existir; então nenhum tema é excluído e há aviso).
        n: número de trechos.
        seed: semente.
        destino: se informado, grava os textos separados por linha em branco.

    Returns:
        Trechos escolhidos, em ordem de id.
    """
    if gold_test is None:
        log.warning(
            "gold de teste ainda não existe; a reserva para perplexidade não exclui temas do teste. "
            "Depois da divisão, rode `qa-manual importar-revisao --dividir --reservar-ppl %d`",
            n,
        )
    temas_teste = {it.get("tema_id", "") for it in gold_test or []}
    candidatos = [t for t in trechos if t.tema_id not in temas_teste]
    if len(candidatos) < n:
        log.warning("só %d trechos fora dos temas do teste (pedidos %d)", len(candidatos), n)
    escolhidos = sorted(random.Random(seed).sample(candidatos, min(n, len(candidatos))), key=lambda t: t.id)
    if destino is not None:
        texto = "\n\n".join(t.texto for t in escolhidos) + "\n"
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        Path(destino).write_text(texto, encoding="utf-8")
        log.info("perplexidade: %d trechos reservados (sha256 %s)", len(escolhidos), sha256_texto(texto)[:12])
    return escolhidos


def validar_gold(itens: list[dict], origem: str | Path) -> None:
    """Confere campos obrigatórios e tipos de um arquivo gold.

    Raises:
        ErroUsuario: se o arquivo estiver vazio ou algum item for inválido (a mensagem cita só ids).
    """
    if not itens:
        raise ErroUsuario(f"{origem}: gold vazio.")
    erros, vistos = [], set()
    for n, it in enumerate(itens, start=1):
        rotulo = it.get("id") or f"item {n}"
        faltando = [c for c in CAMPOS_GOLD if c not in it]
        if faltando:
            erros.append(f"{rotulo}: campo(s) ausente(s): {', '.join(faltando)}")
            continue
        if it["tipo"] not in TIPOS:
            erros.append(f"{rotulo}: tipo '{it['tipo']}' inválido; use {', '.join(TIPOS)}")
        if not isinstance(it["evidencia"], list):
            erros.append(f"{rotulo}: 'evidencia' deve ser uma lista")
        if it["id"] in vistos:
            erros.append(f"{rotulo}: id repetido")
        vistos.add(it["id"])
    if erros:
        raise ErroUsuario(f"{origem}: gold inválido:\n  - " + "\n  - ".join(erros[:20]))
