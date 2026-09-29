"""Interface de linha de comando. Uso: python -m src.cli <comando> [opções]

Argumentos de arquivo omitidos são solicitados interativamente; nenhum caminho de dados é assumido.
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from collections import Counter
from itertools import combinations
from pathlib import Path

from . import analysis, gold as gold_mod
from .evaluate import (
    amostrar_gold,
    concordancia_juiz,
    executar_avaliacao,
    tabela_metricas,
    validar_gold,
)
from .generate import responder
from .index import Indice, indexar_manual, indice_atualizado, verificar_manifesto
from .ollama_client import OllamaClient
from .retrieve import MODOS, Recuperador
from .utils import (
    DIR_INDEX,
    ErroUsuario,
    anexar_jsonl,
    carregar_config,
    carregar_config_indexacao,
    confirmar,
    escrever_json,
    escrever_jsonl,
    ler_json,
    ler_jsonl,
    novo_run_dir,
    solicitar,
    solicitar_arquivo,
    tabela_markdown,
)


def _trecho(texto: str, n: int = 160) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= n else texto[: n - 1] + "…"


def _dir_run(caminho: str | None) -> Path:
    run_dir = Path(solicitar(caminho, "Diretório da execução (runs/<timestamp>)"))
    if not (run_dir / "respostas.jsonl").exists():
        raise ErroUsuario(f"{run_dir}/respostas.jsonl não encontrado. Rode `avaliar` antes.")
    return run_dir


def _preparar_recuperacao(cfg: dict, dir_indice: Path | None = None) -> tuple[OllamaClient, Recuperador | None]:
    cliente = OllamaClient(cfg.get("ollama_url", "http://localhost:11434"))
    if cfg["modo"] == "nenhum":
        return cliente, None
    dir_indice = Path(dir_indice or cfg.get("indice", DIR_INDEX))
    verificar_manifesto(dir_indice, cfg.get("embeddings"))
    return cliente, Recuperador(Indice(dir_indice), cliente, cfg.get("rrf_k", 60))


def _confirmar_sobrescrita(caminhos: list[Path], forcar: bool) -> None:
    existentes = [c for c in caminhos if c.exists()]
    if existentes and not forcar:
        if not confirmar(f"{', '.join(map(str, existentes))} já existe(m). Sobrescrever?"):
            raise ErroUsuario("Operação cancelada para não sobrescrever arquivos existentes (use --sobrescrever).")


# ---------------------------------------------------------------------------
# Etapas 1 e 2
# ---------------------------------------------------------------------------

def cmd_indexar(a: argparse.Namespace) -> None:
    manual = solicitar_arquivo(a.manual, "Caminho do manual (PDF)", (".pdf",))
    params = carregar_config_indexacao()
    if a.tamanho:
        params["tamanho_tokens"] = a.tamanho
    if a.sobreposicao is not None:
        params["sobreposicao_tokens"] = a.sobreposicao
    if a.sem_tabelas:
        params["detectar_tabelas"] = False
    destino = Path(a.indice)
    cliente = OllamaClient(params.get("ollama_url", "http://localhost:11434"))

    print(f"Indexando {manual} → {destino}/ (passagens de {params['tamanho_tokens']} tokens, "
          f"sobreposição {params['sobreposicao_tokens']}, embeddings {params['embeddings']})")
    estat, manifesto = indexar_manual(manual, params, destino, cliente)
    t = estat["tamanho_passagem"]
    print(textwrap.dedent(f"""
        Páginas: {estat['paginas']} ({estat['paginas_sem_texto']} sem texto) | palavras: {estat['palavras']}
        Seções: {estat['secoes']} | passagens: {estat['passagens']} | tabelas: {estat['tabelas_detectadas']}
        Palavras por passagem: min {t['min']}, mediana {t['mediana']}, média {t['media']}, máx {t['max']} (limite {estat['limite_palavras_passagem']})
        Cabeçalhos/rodapés removidos: {estat['linhas_cabecalho_rodape_removidas']} linhas
        Arquivos: {destino}/passages.jsonl, stats.json, bm25/, dense.faiss, dense_ids.json, manifest.json
        Próximo passo: python -m src.cli inspecionar --n 20"""))


def cmd_inspecionar(a: argparse.Namespace) -> None:
    dir_indice = Path(a.indice)
    arq = dir_indice / "passages.jsonl"
    if not arq.exists():
        raise ErroUsuario(f"{arq} não encontrado. Rode `indexar` antes.")
    passagens = ler_jsonl(arq)
    amostra = random.Random(a.seed).sample(passagens, min(a.n, len(passagens)))
    for p in sorted(amostra, key=lambda x: x["id"]):
        print("=" * 100)
        print(f"{p['id']} | seção: {p['secao']} | p. {p['pagina']}-{p.get('pagina_fim', p['pagina'])} | {p['n_palavras']} palavras")
        print(textwrap.indent(p["texto"], "    "))
    estat = ler_json(dir_indice / "stats.json") if (dir_indice / "stats.json").exists() else {}
    print("=" * 100)
    print(f"Verifique: texto legível, seção e página corretas. Vazias: {estat.get('passagens_vazias')}, "
          f"acima do limite: {estat.get('acima_do_limite')}.")


def cmd_testar_indice(a: argparse.Namespace) -> None:
    """Critério de aceite da etapa 2: consultas escritas à mão devem trazer a passagem esperada no top-k."""
    arq = solicitar_arquivo(a.consultas, "Arquivo de consultas de teste (.jsonl com pergunta e esperada)", (".jsonl",))
    consultas = ler_jsonl(arq)
    cfg = carregar_config("S3")
    _, rec = _preparar_recuperacao(cfg, Path(a.indice))
    linhas, acertos = [], {m: 0 for m in ("bm25", "denso")}
    for c in consultas:
        esperadas = set(c["esperada"] if isinstance(c["esperada"], list) else [c["esperada"]])
        linha = [_trecho(c["pergunta"], 60)]
        for modo in ("bm25", "denso"):
            ids = [pid for pid, _ in rec.recuperar(c["pergunta"], modo, a.k)]
            pos = next((i for i, pid in enumerate(ids, 1) if pid in esperadas), None)
            acertos[modo] += pos is not None
            linha.append(pos or "—")
        linhas.append(linha)
    print(tabela_markdown(["Consulta", f"Posição BM25 (top-{a.k})", f"Posição denso (top-{a.k})"], linhas))
    for modo, n in acertos.items():
        print(f"{modo}: {n}/{len(consultas)} consultas com a passagem esperada no top-{a.k}")


# ---------------------------------------------------------------------------
# Etapas 3 e 4
# ---------------------------------------------------------------------------

def cmd_recuperar(a: argparse.Namespace) -> None:
    pergunta = solicitar(a.pergunta, "Pergunta")
    modo = solicitar(a.modo, f"Modo ({', '.join(MODOS)})")
    if modo not in MODOS:
        raise ErroUsuario(f"Modo inválido: {modo}. Use {', '.join(MODOS)}.")
    cfg = carregar_config(a.config)
    k = a.k or cfg.get("k", 5)
    _, rec = _preparar_recuperacao({**cfg, "modo": modo}, Path(a.indice) if a.indice else None)
    resultados = rec.recuperar(pergunta, modo, k)
    linhas = []
    for pos, (pid, score) in enumerate(resultados, 1):
        p = rec.indice.por_id[pid]
        linhas.append([pos, pid, round(score, 4), _trecho(p["secao"], 40), p["pagina"], _trecho(p["texto"], 70)])
    print(tabela_markdown(["#", "id", "score", "seção", "página", "trecho"], linhas))


def _imprimir_resposta(reg: dict) -> None:
    print(f"\nResposta: {reg['resposta']}")
    if reg["recuperadas"]:
        print("Passagens usadas:")
        for i, r in enumerate(reg["recuperadas"], 1):
            print(f"  [{i}] {r['id']} (score {r['score']:.4f}) seção {r['secao']}, p. {r['pagina']}")
    print(f"Latência: {reg['latencia_s']:.1f} s | tokens: prompt {reg['tokens_prompt']}, "
          f"resposta {reg['tokens_resposta']} | abstenção: {'sim' if reg['absteve'] else 'não'}")


def cmd_perguntar(a: argparse.Namespace) -> None:
    cfg = carregar_config(solicitar(a.config, "Configuração (S0, S1, S2 ou S3)"))
    if a.arquivo:
        arq = solicitar_arquivo(a.arquivo, "Arquivo de perguntas (.txt)", (".txt",))
        perguntas = [l.strip() for l in arq.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not perguntas:
            raise ErroUsuario(f"{arq} não contém perguntas.")
    elif a.pergunta:
        perguntas = [a.pergunta]
    else:
        perguntas = None  # modo interativo
    cliente, rec = _preparar_recuperacao(cfg)

    run_id, run_dir = None, None

    def registrar(n: int, pergunta: str) -> None:
        nonlocal run_id, run_dir
        reg = responder(pergunta, cfg, rec, cliente)
        if run_dir is None:
            run_id, run_dir = novo_run_dir(f"perguntar_{cfg['nome']}")
        anexar_jsonl(run_dir / "respostas.jsonl", {"run_id": run_id, "q_id": f"livre-{n:03d}", **reg})
        if perguntas is not None and len(perguntas) > 1:
            print(f"\n[{n}/{len(perguntas)}] {pergunta}")
        _imprimir_resposta(reg)

    if perguntas is not None:
        for n, q in enumerate(perguntas, 1):
            registrar(n, q)
    else:
        if not sys.stdin.isatty():
            raise ErroUsuario("Informe --pergunta ou --arquivo (sem terminal interativo).")
        print(f"Configuração {cfg['nome']} ({cfg['modo']}, leitor {cfg['leitor']}). Linha vazia para sair.")
        n = 0
        while True:
            try:
                q = input("\nPergunta: ").strip()
            except EOFError:
                break
            if not q:
                break
            n += 1
            registrar(n, q)
    if run_dir:
        print(f"\nRegistro: {run_dir}/respostas.jsonl")


# ---------------------------------------------------------------------------
# Etapa 5
# ---------------------------------------------------------------------------

def cmd_gerar_perguntas(a: argparse.Namespace) -> None:
    dir_indice = Path(a.indice)
    verificar_manifesto(dir_indice)
    n = int(solicitar(str(a.n) if a.n else None, "Número de candidatas (ex.: 150)"))
    if n <= 0:
        raise ErroUsuario("O número de candidatas deve ser positivo.")
    saida = Path(a.saida)
    _confirmar_sobrescrita([saida], a.sobrescrever)
    params = carregar_config_indexacao()
    modelo = a.modelo or params.get("juiz", "llama3.1:8b")
    cliente = OllamaClient(params.get("ollama_url", "http://localhost:11434"))
    indice = Indice(dir_indice, carregar_denso=False)
    rec = Recuperador(indice)
    print(f"Gerando {n} candidatas com {modelo} a partir de {len(indice.passagens)} passagens...")
    candidatas = gold_mod.gerar_candidatas(indice.passagens, n, cliente, modelo, seed=a.seed, recuperador_bm25=rec)
    escrever_jsonl(saida, candidatas)
    tipos = Counter(c["tipo"] for c in candidatas)
    secoes = len({c["secao"] for c in candidatas if c["secao"]})
    print(f"{len(candidatas)} candidatas gravadas em {saida} | tipos: {dict(tipos)} | seções cobertas: {secoes}")
    print(f"Próximo passo: python -m src.cli revisar --candidatas {saida} --anotador <nome>")


def cmd_revisar(a: argparse.Namespace) -> None:
    arq = solicitar_arquivo(a.candidatas, "Arquivo de candidatas (.jsonl)", (".jsonl",))
    anotador = solicitar(a.anotador, "Nome do anotador")
    candidatas = ler_jsonl(arq)
    passagens = {p["id"]: p for p in ler_jsonl(Path(a.indice) / "passages.jsonl")} if (Path(a.indice) / "passages.jsonl").exists() else {}
    somente = gold_mod.subconjunto_comum([c["id"] for c in candidatas], a.fracao_comum) if a.comum else None
    destino = Path(a.dir_revisoes) / f"{anotador}.jsonl"
    gold_mod.revisar(candidatas, passagens, destino, anotador, somente)


def cmd_concordancia(a: argparse.Namespace) -> None:
    arquivos = a.revisoes or solicitar(None, "Dois arquivos de revisão (separados por espaço)").split()
    if len(arquivos) != 2:
        raise ErroUsuario("Informe exatamente dois arquivos de revisão.")
    revs = [ler_jsonl(solicitar_arquivo(f, "Revisão", (".jsonl",))) for f in arquivos]
    resultado = gold_mod.concordancia_anotadores(*revs)
    saida = Path(a.saida) if a.saida else Path(arquivos[0]).parent / "concordancia.json"
    escrever_json(saida, resultado)
    print(tabela_markdown(
        ["Aspecto", "Concordância", "Kappa"],
        [
            ["Manter/descartar", resultado["decisao_manter"]["concordancia"], resultado["decisao_manter"]["kappa"]],
            ["Tipo", resultado["tipo"]["concordancia"], resultado["tipo"]["kappa"]],
            ["Resposta (F1 médio)", resultado["resposta_f1_medio"], None],
            ["Mesma evidência", resultado["evidencia_igual"], None],
        ],
    ))
    print(f"\nItens em comum: {resultado['itens_comuns']} | gravado em {saida}")


def cmd_dividir_gold(a: argparse.Namespace) -> None:
    arq = solicitar_arquivo(a.candidatas, "Arquivo de candidatas (.jsonl)", (".jsonl",))
    arquivos = a.revisoes or solicitar(None, "Arquivos de revisão (prioridade na ordem, separados por espaço)").split()
    revisoes = [ler_jsonl(solicitar_arquivo(f, "Revisão", (".jsonl",))) for f in arquivos]
    passagens = ler_jsonl(Path(a.indice) / "passages.jsonl")
    secao_de = {p["id"]: p["secao"] for p in passagens}
    itens, contagem = gold_mod.consolidar(ler_jsonl(arq), revisoes)
    if not itens:
        raise ErroUsuario("Nenhum item aprovado nas revisões.")
    desconhecidas = {e for it in itens for e in it["evidencia"] if e not in secao_de}
    if desconhecidas:
        print(f"Aviso: evidências fora do índice atual: {sorted(desconhecidas)[:10]}", file=sys.stderr)
    dev, teste = gold_mod.dividir_por_secao(itens, secao_de, a.frac_dev, a.seed)
    gold_dev, gold_test = gold_mod.montar_gold(itens, dev, teste)
    saida_dev, saida_test = Path(a.saida_dev), Path(a.saida_test)
    _confirmar_sobrescrita([saida_dev, saida_test], a.sobrescrever)
    escrever_jsonl(saida_dev, gold_dev)
    escrever_jsonl(saida_test, gold_test)
    resumo = gold_mod.resumo_divisao(gold_dev, gold_test, secao_de)
    print(f"Revisões: {contagem}")
    print(tabela_markdown(
        ["Split", "Itens", "factual", "procedimental", "sem_resposta", "Seções"],
        [[s, resumo[s]["itens"], *(resumo[s]["tipos"].get(t, 0) for t in ("factual", "procedimental", "sem_resposta")),
          resumo[s]["secoes"]] for s in ("dev", "test")],
    ))
    if resumo["secoes_em_ambos"]:
        print(f"ATENÇÃO: seções nos dois conjuntos: {resumo['secoes_em_ambos']}")
    else:
        print("Nenhuma seção aparece nos dois conjuntos.")
    for s in ("dev", "test"):
        faltando = [t for t in ("factual", "procedimental", "sem_resposta") if not resumo[s]["tipos"].get(t)]
        if faltando:
            print(f"Aviso: {s} não tem itens do(s) tipo(s) {faltando}.")
    print(f"Gravados: {saida_dev} ({len(gold_dev)}) e {saida_test} ({len(gold_test)})")


# ---------------------------------------------------------------------------
# Etapa 6
# ---------------------------------------------------------------------------

def _configs(nomes: list[str] | None) -> list[dict]:
    nomes = nomes or solicitar(None, "Configurações (ex.: S0 S1 S2 S3)").split()
    return [carregar_config(n) for n in nomes]


def _carregar_gold(caminho: str | None, mensagem: str = "Arquivo gold (.jsonl)") -> tuple[Path, list[dict]]:
    arq = solicitar_arquivo(caminho, mensagem, (".jsonl",))
    gold = ler_jsonl(arq)
    if not gold:
        raise ErroUsuario(f"{arq} está vazio.")
    validar_gold(gold, str(arq))
    return arq, gold


def cmd_avaliar(a: argparse.Namespace) -> None:
    arq_gold, gold = _carregar_gold(a.gold)
    configs = _configs(a.configs)
    if a.retomar:
        run_dir = Path(a.retomar)
        if not run_dir.is_dir():
            raise ErroUsuario(f"Diretório de execução não encontrado: {run_dir}")
        run_id = run_dir.name
    else:
        run_id, run_dir = novo_run_dir("avaliar")
    gold = amostrar_gold(gold, a.limite)
    print(f"Avaliando {len(gold)} itens de {arq_gold} em {', '.join(c['nome'] for c in configs)} "
          f"× {a.repeticoes} repetição(ões) → {run_dir}/")
    metrics = executar_avaliacao(
        gold, configs, a.repeticoes, run_dir, run_id, usar_juiz=not a.sem_juiz, k_override=a.k,
        meta_extra={"gold": str(arq_gold), "limite": a.limite},
    )
    print("\n" + tabela_metricas(metrics))
    print(f"\nArquivos: {run_dir}/respostas.jsonl, {run_dir}/metrics.json")


def cmd_validar_juiz(a: argparse.Namespace) -> None:
    run_dir = _dir_run(a.run)
    registros = ler_jsonl(run_dir / "respostas.jsonl")
    # Só itens em que o juiz de fato foi consultado (respondíveis e sem abstenção).
    elegiveis = [r for r in registros if r.get("nota_juiz") is not None and r["tipo"] != "sem_resposta" and not r["absteve"]]
    if not elegiveis:
        raise ErroUsuario("Não há itens julgados pelo modelo nesta execução (rode avaliar sem --sem-juiz).")
    amostra = random.Random(a.seed).sample(elegiveis, min(a.n, len(elegiveis)))
    arq = run_dir / "juiz_manual.jsonl"
    feitos = {(m["config"], m["rep"], m["q_id"]): m for m in ler_jsonl(arq)} if arq.exists() else {}
    fila = [r for r in amostra if (r["config"], r["rep"], r["q_id"]) not in feitos]
    if fila and sys.stdin.isatty():
        print(f"{len(fila)} itens para julgar à mão (0 incorreta, 1 parcial, 2 correta; 's' sai). A nota do juiz fica oculta.")
        for n, r in enumerate(fila, 1):
            print("=" * 100)
            print(f"({n}/{len(fila)}) {r['config']} {r['q_id']}\n  PERGUNTA:   {r['pergunta']}\n"
                  f"  REFERÊNCIA: {r['resposta_ref']}\n  SISTEMA:    {r.get('resposta_curta') or r['resposta']}")
            while True:
                op = input("  Nota > ").strip().lower()
                if op in {"0", "1", "2", "s"}:
                    break
            if op == "s":
                break
            m = {"config": r["config"], "rep": r["rep"], "q_id": r["q_id"], "nota_manual": int(op), "nota_juiz": r["nota_juiz"]}
            anexar_jsonl(arq, m)
            feitos[(r["config"], r["rep"], r["q_id"])] = m
    elif fila:
        print(f"{len(fila)} itens pendentes; rode em um terminal para julgar.", file=sys.stderr)
    pares = [(m["nota_manual"], m["nota_juiz"]) for m in feitos.values()]
    if not pares:
        return
    resultado = concordancia_juiz(pares)
    arq_metrics = run_dir / "metrics.json"
    metrics = ler_json(arq_metrics) if arq_metrics.exists() else {}
    metrics["_validacao_juiz"] = resultado
    escrever_json(arq_metrics, metrics)
    print(tabela_markdown(["Medida", "Valor"], [[k, v] for k, v in resultado.items()]))
    print(f"Registrado em {arq_metrics} (_validacao_juiz).")


# ---------------------------------------------------------------------------
# Etapa 7
# ---------------------------------------------------------------------------

def cmd_varrer(a: argparse.Namespace) -> None:
    arq_gold, gold = _carregar_gold(a.gold, "Arquivo gold de desenvolvimento (.jsonl)")
    if "test" in arq_gold.name and not confirmar("O arquivo parece ser de teste; a varredura deve usar o dev. Continuar?"):
        raise ErroUsuario("Varredura cancelada.")
    manual = solicitar_arquivo(a.manual, "Caminho do manual (PDF)", (".pdf",))
    cfg = carregar_config(a.config)
    params = {**carregar_config_indexacao(), "embeddings": cfg["embeddings"]}
    base_dir = Path(a.indice)
    base = {p["id"]: p for p in ler_jsonl(base_dir / "passages.jsonl")}
    cliente = OllamaClient(cfg.get("ollama_url", "http://localhost:11434"))
    gold = amostrar_gold(gold, a.limite)
    if a.retomar:
        run_dir = Path(a.retomar)
        run_id = run_dir.name
    else:
        run_id, run_dir = novo_run_dir("varredura")

    linhas, resultados = [], []
    for t in a.tamanhos:
        p = {**params, "tamanho_tokens": t}
        destino = base_dir if indice_atualizado(base_dir, manual, p) else base_dir / f"t{t}"
        if not indice_atualizado(destino, manual, p):
            print(f"Indexando com passagens de {t} tokens → {destino}/")
            indexar_manual(manual, p, destino, cliente)
        mapa = analysis.mapear_evidencias(gold, base, ler_jsonl(destino / "passages.jsonl"))
        sem_mapa = sum(1 for g in gold if g["evidencia"] and not mapa[g["id"]])
        if sem_mapa:
            print(f"Aviso: {sem_mapa} itens sem evidência correspondente no índice t{t}.", file=sys.stderr)
        for k in a.ks:
            nome = f"{cfg['nome']}_t{t}_k{k}"
            sub = run_dir / f"t{t}_k{k}"
            sub.mkdir(parents=True, exist_ok=True)
            print(f"→ {nome}")
            m = executar_avaliacao(
                gold, [{**cfg, "nome": nome, "indice": str(destino)}], 1, sub, run_id,
                usar_juiz=False, dir_indice=destino, k_override=k, mapa_evidencias=mapa,
                meta_extra={"gold": str(arq_gold), "tamanho_tokens": t},
            )[nome]
            recall = next((v for c, v in m.items() if c.startswith("recall@")), None)
            resultados.append({"tamanho_tokens": t, "k": k, "recall@k": recall, "mrr": m["mrr"], "f1": m["f1"],
                               "f1_respondiveis": m["f1_respondiveis"], "abst_indevida": m["abst_indevida"],
                               "latencia_mediana_s": m["latencia_mediana_s"]})
            linhas.append([t, k, recall, m["mrr"], m["f1"], m["f1_respondiveis"], m["abst_indevida"], m["latencia_mediana_s"]])

    melhor = max(resultados, key=lambda r: (r["f1"] or 0, r["recall@k"] or 0, -(r["latencia_mediana_s"] or 0)))
    escrever_json(run_dir / "varredura.json", {"gold": str(arq_gold), "config": cfg["nome"], "resultados": resultados, "melhor": melhor})
    print("\n" + tabela_markdown(["Tamanho (tokens)", "k", "Recall@k", "MRR", "F1", "F1 respondíveis", "Abst. indevida", "Lat. mediana (s)"], linhas))
    print(f"\nMelhor (F1, desempate por Recall@k): tamanho {melhor['tamanho_tokens']} tokens, k = {melhor['k']}.")
    print("Para usar: ajuste tamanho_tokens em configs/indexacao.yaml e k em configs/base.yaml, e reindexe.")
    print(f"Resultados: {run_dir}/varredura.json")


def cmd_analisar(a: argparse.Namespace) -> None:
    run_dir = _dir_run(a.run)
    registros = ler_jsonl(run_dir / "respostas.jsonl")
    metrics = ler_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    meta = metrics.get("_meta", {})
    configs = meta.get("configs") or sorted({r["config"] for r in registros})
    partes = [f"# Análise da execução {run_dir.name}\n", "## Métricas (média ± desvio entre repetições)\n",
              tabela_metricas(metrics) if metrics else "(metrics.json ausente)"]

    if "_validacao_juiz" in metrics:
        v = metrics["_validacao_juiz"]
        partes.append(f"\nValidação do juiz ({v['n']} itens): concordância exata {v['concordancia_exata']}, "
                      f"kappa ponderado {v['kappa_ponderado_quadratico']}.")

    # Comparações pareadas
    # Sem notas do juiz, o critério "juiz" recai em F1 >= 0,5 (ver analysis.acerto_binario).
    criterio = a.criterio
    comps = [analysis.comparar(registros, x, y, criterio) for x, y in combinations(configs, 2)]
    escrever_json(run_dir / "comparacoes.json", comps)
    partes += ["\n## Comparação pareada por pergunta\n",
               f"Δ F1 com IC 95% por bootstrap pareado (10.000 reamostragens); McNemar exato sobre acerto binário "
               f"(critério: {criterio}; maioria entre repetições).\n", analysis.tabela_comparacoes(comps)]

    # Curva Recall@k
    if not a.sem_curva:
        arq_gold = a.gold or meta.get("gold")
        arq_gold, gold = _carregar_gold(arq_gold, "Arquivo gold para a curva Recall@k (.jsonl)")
        cfg = carregar_config(a.config_curva)
        _, rec = _preparar_recuperacao({**cfg, "modo": "hibrido"}, Path(meta["indice"]) if meta.get("indice") else None)
        curva = analysis.curva_recall(gold, rec, a.ks)
        csv = ["modo," + ",".join(f"recall@{k}" for k in a.ks)]
        csv += [f"{m}," + ",".join(str(v[k]) for k in a.ks) for m, v in curva.items()]
        (run_dir / "recall_k.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
        png = analysis.grafico_recall(curva, run_dir / "recall_k.png")
        partes += ["\n## Recall@k por modo de recuperação\n",
                   tabela_markdown(["Modo"] + [f"@{k}" for k in a.ks], [[m] + [v[k] for k in a.ks] for m, v in curva.items()])]
        if png:
            partes.append(f"\n![Recall@k]({png.name})")

    # Erros
    cfg_erros = a.config_erros or (configs[-1] if configs else None)
    arq_amostra = run_dir / "erros_amostra.jsonl"
    if cfg_erros and (not arq_amostra.exists() or a.refazer_amostra):
        amostra = analysis.amostrar_erros(registros, cfg_erros, a.n_erros, a.seed)
        escrever_jsonl(arq_amostra, amostra)
    arq_class = run_dir / "erros_classificados.jsonl"
    partes.append(f"\n## Categorias de erro ({cfg_erros})\n")
    if arq_class.exists():
        partes.append(analysis.tabela_categorias(ler_jsonl(arq_class)))
    elif arq_amostra.exists():
        partes.append("Categorias **sugeridas** automaticamente (confirme com `classificar-erros`):\n\n"
                      + analysis.tabela_categorias(ler_jsonl(arq_amostra)))

    texto = "\n".join(partes) + "\n"
    (run_dir / "analise.md").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Gravado em {run_dir}/analise.md")


def cmd_classificar_erros(a: argparse.Namespace) -> None:
    run_dir = _dir_run(a.run)
    arq = run_dir / "erros_amostra.jsonl"
    if not arq.exists():
        raise ErroUsuario(f"{arq} não encontrado. Rode `analisar --run {run_dir}` antes.")
    meta = ler_json(run_dir / "metrics.json").get("_meta", {}) if (run_dir / "metrics.json").exists() else {}
    arq_p = Path(meta.get("indice") or DIR_INDEX) / "passages.jsonl"
    passagens = {p["id"]: p for p in ler_jsonl(arq_p)} if arq_p.exists() else {}
    analysis.classificar_interativo(ler_jsonl(arq), run_dir / "erros_classificados.jsonl", passagens)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description="QA sobre manual com LLM local (Ollama).")
    sub = ap.add_subparsers(dest="comando", required=True, metavar="comando")

    p = sub.add_parser("indexar", help="Etapas 1-2: PDF → passagens → índices BM25 e denso")
    p.add_argument("--manual", help="caminho do manual em PDF")
    p.add_argument("--indice", default=str(DIR_INDEX), help="diretório do índice (padrão: index)")
    p.add_argument("--tamanho", type=int, help="tamanho da passagem em tokens (padrão: configs/indexacao.yaml)")
    p.add_argument("--sobreposicao", type=int, help="sobreposição em tokens")
    p.add_argument("--sem-tabelas", action="store_true", help="desativa a detecção de tabelas")
    p.set_defaults(func=cmd_indexar)

    p = sub.add_parser("inspecionar", help="Etapa 1: mostra passagens aleatórias para inspeção manual")
    p.add_argument("--indice", default=str(DIR_INDEX))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_inspecionar)

    p = sub.add_parser("testar-indice", help="Etapa 2: consultas manuais devem achar a passagem esperada")
    p.add_argument("--consultas", help="JSONL com {\"pergunta\": ..., \"esperada\": \"p0042\"}")
    p.add_argument("--indice", default=str(DIR_INDEX))
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(func=cmd_testar_indice)

    p = sub.add_parser("recuperar", help="Etapa 3: depuração da recuperação")
    p.add_argument("--pergunta")
    p.add_argument("--modo", choices=MODOS)
    p.add_argument("--k", type=int)
    p.add_argument("--config", default="S3", help="configuração de onde ler k e rrf_k (padrão: S3)")
    p.add_argument("--indice")
    p.set_defaults(func=cmd_recuperar)

    p = sub.add_parser("perguntar", help="Etapa 4: responde perguntas (única, arquivo ou interativo)")
    p.add_argument("--config", help="S0, S1, S2, S3 ou caminho de YAML")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pergunta")
    g.add_argument("--arquivo", help="arquivo .txt com uma pergunta por linha")
    p.set_defaults(func=cmd_perguntar)

    p = sub.add_parser("gerar-perguntas", help="Etapa 5: gera candidatas de perguntas com o LLM local")
    p.add_argument("--n", type=int, help="número de candidatas (ex.: 150)")
    p.add_argument("--indice", default=str(DIR_INDEX))
    p.add_argument("--saida", default="data/gold_candidatas.jsonl")
    p.add_argument("--modelo", help="modelo gerador (padrão: juiz de configs/base.yaml)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sobrescrever", action="store_true")
    p.set_defaults(func=cmd_gerar_perguntas)

    p = sub.add_parser("revisar", help="Etapa 5: revisão manual interativa das candidatas")
    p.add_argument("--candidatas")
    p.add_argument("--anotador")
    p.add_argument("--comum", action="store_true", help="revisa só o subconjunto comum (para concordância)")
    p.add_argument("--fracao-comum", type=float, default=0.3)
    p.add_argument("--indice", default=str(DIR_INDEX))
    p.add_argument("--dir-revisoes", default="data/revisoes")
    p.set_defaults(func=cmd_revisar)

    p = sub.add_parser("concordancia", help="Etapa 5: concordância entre dois anotadores")
    p.add_argument("--revisoes", nargs="+")
    p.add_argument("--saida")
    p.set_defaults(func=cmd_concordancia)

    p = sub.add_parser("dividir-gold", help="Etapa 5: consolida revisões e divide dev/teste por seção")
    p.add_argument("--candidatas")
    p.add_argument("--revisoes", nargs="+", help="arquivos de revisão, em ordem de prioridade")
    p.add_argument("--indice", default=str(DIR_INDEX))
    p.add_argument("--frac-dev", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saida-dev", default="data/gold_dev.jsonl")
    p.add_argument("--saida-test", default="data/gold_test.jsonl")
    p.add_argument("--sobrescrever", action="store_true")
    p.set_defaults(func=cmd_dividir_gold)

    p = sub.add_parser("avaliar", help="Etapa 6: executa configurações sobre o gold e calcula métricas")
    p.add_argument("--gold")
    p.add_argument("--configs", nargs="+")
    p.add_argument("--repeticoes", type=int, default=1)
    p.add_argument("--limite", type=int, help="avalia só N itens (amostra fixa), ex.: experimento mínimo")
    p.add_argument("--k", type=int, help="sobrescreve o k das configurações")
    p.add_argument("--sem-juiz", action="store_true")
    p.add_argument("--retomar", help="diretório runs/<timestamp> de uma execução interrompida")
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("validar-juiz", help="Etapa 6: julgamento manual de 50 itens e concordância com o juiz")
    p.add_argument("--run")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_validar_juiz)

    p = sub.add_parser("varrer", help="Etapa 7: varre tamanho de passagem e k no conjunto de desenvolvimento")
    p.add_argument("--gold")
    p.add_argument("--manual")
    p.add_argument("--config", default="S3")
    p.add_argument("--tamanhos", type=int, nargs="+", default=[300, 400, 500])
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5, 8])
    p.add_argument("--indice", default=str(DIR_INDEX), help="índice base, ao qual as evidências do gold se referem")
    p.add_argument("--limite", type=int)
    p.add_argument("--retomar")
    p.set_defaults(func=cmd_varrer)

    p = sub.add_parser("analisar", help="Etapa 7: comparações pareadas, curva Recall@k e erros")
    p.add_argument("--run")
    p.add_argument("--gold", help="padrão: o gold usado na execução")
    p.add_argument("--criterio", choices=["juiz", "em", "f1"], default="juiz")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20])
    p.add_argument("--config-curva", default="S3")
    p.add_argument("--sem-curva", action="store_true")
    p.add_argument("--config-erros", help="configuração da amostra de erros (padrão: a última)")
    p.add_argument("--n-erros", type=int, default=50)
    p.add_argument("--refazer-amostra", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_analisar)

    p = sub.add_parser("classificar-erros", help="Etapa 7: classificação manual interativa da amostra de erros")
    p.add_argument("--run")
    p.set_defaults(func=cmd_classificar_erros)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    try:
        args.func(args)
    except ErroUsuario as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
