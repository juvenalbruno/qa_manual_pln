"""Interface de linha de comando ``qa-manual``.

Todo comando aceita ``--config`` (padrão ``configs/base.yaml``) e ``--sobrescrever``. Arquivos de entrada
omitidos são pedidos interativamente; arquivo inexistente encerra com código 2.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
from pathlib import Path

from . import __version__, config, gold, indexacao
from .io_utils import (
    ErroUsuario,
    agora_iso,
    anexar_jsonl,
    anexar_log_arquivo,
    criar_run_dir,
    escrever_json,
    escrever_jsonl,
    ler_json,
    ler_jsonl,
    novo_run_id,
    pasta_sincronizada,
    remover_log_arquivo,
    sha256_arquivo,
    validar_arquivo,
    versao_codigo,
)
from .ollama_client import ModeloAusente, criar_cliente, instrucao_modelo, modelo_presente

log = logging.getLogger("qa_manual.cli")


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def _pedir_arquivo(
    valor: str | None, mensagem: str, extensoes: tuple[str, ...] = (), padrao: Path | None = None
) -> Path:
    """Arquivo informado por argumento ou, se omitido, pedido com ``input()`` (validado em seguida)."""
    if not valor:
        if sys.stdin.isatty():
            sugestao = f" [{padrao}]" if padrao else ""
            try:
                valor = input(f"{mensagem}{sugestao}: ").strip() or (str(padrao) if padrao else "")
            except EOFError:
                valor = ""
        elif padrao is not None:
            valor = str(padrao)
            print(f"{mensagem}: usando {valor}", file=sys.stderr)
        if not valor:
            raise ErroUsuario(f"{mensagem}: nenhum arquivo informado.")
    return validar_arquivo(valor, extensoes)


def _nao_sobrescrever(caminhos: list[Path], forcar: bool) -> None:
    existentes = [str(p) for p in caminhos if Path(p).exists()]
    if existentes and not forcar:
        raise ErroUsuario(f"{', '.join(existentes)} já existe(m); use --forcar para sobrescrever.")


def _cfg(a: argparse.Namespace) -> config.Config:
    return config.carregar(a.config, a.sobrescrever)


def _dir_dados(cfg: config.Config) -> Path:
    return Path(cfg.paths.trechos).parent


def _exigir_modelos(client, modelos: list[str]) -> None:
    disponiveis = client.modelos_disponiveis()
    for m in modelos:
        if not modelo_presente(m, disponiveis):
            raise ModeloAusente(instrucao_modelo(m))


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------


def _proteger_confidencialidade(cfg: config.Config, pdf: Path, permitir: bool) -> None:
    """Recusa gravar dados do manual em pasta sincronizada com a nuvem (regra 0.1)."""
    alvos = {
        "manual": pdf,
        "trechos": Path(cfg.paths.trechos).parent,
        "índice": Path(cfg.paths.index_dir),
        "execuções": Path(cfg.paths.runs_dir),
    }
    sincronizados = [
        f"{rotulo} ({caminho}: {servico})"
        for rotulo, caminho in alvos.items()
        if (servico := pasta_sincronizada(caminho))
    ]
    if not sincronizados:
        return
    msg = (
        "pasta(s) sincronizada(s) com a nuvem: " + "; ".join(sincronizados) + ". O manual e seus derivados "
        "seriam enviados para fora da máquina. Mova o projeto para uma pasta local (ou, no iCloud, use uma pasta "
        "terminada em .nosync)."
    )
    if not permitir:
        raise ErroUsuario(msg + " Para um documento público de teste, use --permitir-pasta-sincronizada.")
    log.warning("%s Prosseguindo por --permitir-pasta-sincronizada.", msg)


def cmd_indexar(a: argparse.Namespace) -> None:
    cfg = _cfg(a)
    pdf = _pedir_arquivo(a.manual, "Manual em PDF", (".pdf",))
    _proteger_confidencialidade(cfg, pdf, a.permitir_pasta_sincronizada)
    mudancas = {}
    if a.max_tokens is not None:
        mudancas["max_tokens"] = a.max_tokens
    if a.sobreposicao_paragrafos is not None:
        mudancas["sobreposicao_paragrafos"] = a.sobreposicao_paragrafos
    if mudancas:
        import dataclasses

        cfg = cfg.com(trechos=dataclasses.replace(cfg.trechos, **mudancas))
    client = criar_cliente(cfg)  # modelo ausente ou Ollama fora do ar viram erro claro no primeiro embed
    t0 = time.perf_counter()
    stats = indexacao.indexar(pdf, cfg, client, reservar_ppl=a.reservar_ppl)
    print(
        f"\n{stats['n_trechos']} trechos em {stats['n_temas']} temas ({stats['n_paginas']} páginas), "
        f"tokens: mediana {stats['tokens']['mediana']}, máximo {stats['tokens']['maximo']}. "
        f"Indexação em {time.perf_counter() - t0:.1f} s."
    )
    print(f"Gravados: {cfg.paths.trechos}, {indexacao.caminho_stats(cfg)}, {cfg.paths.index_dir}/")
    if a.reservar_ppl:
        print(f"Trechos para perplexidade: {cfg.paths.ppl_holdout}")


def cmd_recuperar(a: argparse.Namespace) -> None:
    from .retrieve import recuperar

    cfg = _cfg(a)
    k = a.k or cfg.k
    cfg = cfg.com(retriever=a.modo, k=k, k_candidatos_por_lista=max(k, cfg.k_candidatos_por_lista))
    pergunta = a.pergunta or input("Pergunta: ").strip()
    client = criar_cliente(cfg)
    indices = indexacao.carregar_indices(cfg, client, denso=a.modo in ("denso", "hibrido"))
    recuperadas, tempos = recuperar(pergunta, a.modo, cfg, indices.bm25, indices.denso)
    print(f"modo {a.modo}, k={k}, embedding {tempos['t_embedding_s']:.3f} s, busca {tempos['t_busca_s']:.3f} s")
    for r in recuperadas:
        t = indices.por_id[r["id"]]
        paginas = f"p. {t.pagina_inicio}" + (f"-{t.pagina_fim}" if t.pagina_fim != t.pagina_inicio else "")
        print(
            f"{r['rank']:>2}. {r['id']}  {r['score']:.4f}  seção {t.tema_id or '-'}  {paginas}  "
            f"{t.texto[:100].replace(chr(10), ' ')}..."
        )
    if not recuperadas:
        print("nenhum trecho recuperado")


def _perguntas_do_usuario(a: argparse.Namespace) -> list[str] | None:
    if a.pergunta:
        return [a.pergunta]
    if a.arquivo:
        arq = validar_arquivo(a.arquivo, (".txt",))
        linhas = [ln.strip() for ln in arq.read_text(encoding="utf-8").splitlines()]
        perguntas = [ln for ln in linhas if ln and not ln.startswith("#")]
        if not perguntas:
            raise ErroUsuario(f"{arq}: nenhuma pergunta (uma por linha).")
        return perguntas
    return None  # modo interativo


def cmd_perguntar(a: argparse.Namespace) -> None:
    from .pipeline import perguntar

    cfg = config.carregar_experimento(a.config_exp, a.config, a.sobrescrever)
    perguntas = _perguntas_do_usuario(a)
    client = criar_cliente(cfg)
    _exigir_modelos(client, [cfg.modelo_leitor(a.leitor)])
    indices = None
    if cfg.retriever != "nenhum":
        indices = indexacao.carregar_indices(cfg, client, denso=cfg.retriever in ("denso", "hibrido"))
    run_id = novo_run_id(cfg.nome, a.leitor)
    run_dir = criar_run_dir(cfg.paths.runs_dir, run_id)
    escrever_json(
        run_dir / "config.json",
        {
            "run_id": run_id,
            "criado_em": agora_iso(),
            "versao_codigo": versao_codigo(),
            "leitor": a.leitor,
            "config": cfg.como_dict(),
        },
    )
    handler = anexar_log_arquivo(run_dir / "log.txt")
    try:

        def responder(n: int, pergunta: str) -> None:
            reg = perguntar(pergunta, cfg, a.leitor, indices, client, q_id=f"u{n:03d}", run_id=run_id)
            anexar_jsonl(run_dir / "respostas.jsonl", reg)
            fontes = ", ".join(r["id"] for r in reg["recuperadas"]) or "nenhum trecho (S0)"
            print(f"\nResposta: {reg['resposta']}")
            print(
                f"Trechos: {fontes} | citou fonte: {'sim' if reg['citou_fonte'] else 'não'}, "
                f"fonte válida: {'sim' if reg['fonte_valida'] else 'não'} | {reg['latencia_s']:.1f} s"
            )

        if perguntas is not None:
            for n, p in enumerate(perguntas, 1):
                if len(perguntas) > 1:
                    print(f"\n[{n}/{len(perguntas)}] {p}")
                responder(n, p)
        else:
            print("Modo interativo: digite uma pergunta por linha (linha vazia encerra).")
            n = 0
            while True:
                try:
                    p = input("\nPergunta: ").strip()
                except EOFError:
                    break
                if not p:
                    break
                n += 1
                responder(n, p)
    finally:
        remover_log_arquivo(handler)
    print(f"\nRegistros em {run_dir / 'respostas.jsonl'}")


def cmd_gerar_perguntas(a: argparse.Namespace) -> None:
    cfg = _cfg(a)
    trechos_arq = _pedir_arquivo(a.trechos, "Arquivo de trechos (.jsonl)", (".jsonl",), cfg.paths.trechos)
    saida = Path(a.saida) if a.saida else _dir_dados(cfg) / "gold_candidatas.jsonl"
    _nao_sobrescrever([saida], a.forcar)
    client = criar_cliente(cfg)
    _exigir_modelos(client, [cfg.modelos.gerador])
    candidatas, stats = gold.gerar_candidatas(cfg, client, indexacao.carregar_trechos(trechos_arq), a.n)
    escrever_jsonl(saida, candidatas)
    print(
        f"{stats['geradas']} candidatas de {stats['pedidas']} pedidas; JSON válido em "
        f"{stats['json_valido_pct']:.0%} das {stats['tentativas']} tentativas. Gravado: {saida}"
    )


def cmd_exportar_revisao(a: argparse.Namespace) -> None:
    cfg = _cfg(a)
    cand = _pedir_arquivo(a.candidatas, "Candidatas (.jsonl)", (".jsonl",), _dir_dados(cfg) / "gold_candidatas.jsonl")
    trechos_arq = _pedir_arquivo(a.trechos, "Arquivo de trechos (.jsonl)", (".jsonl",), cfg.paths.trechos)
    saida = Path(a.saida) if a.saida else _dir_dados(cfg) / "gold_revisao.csv"
    _nao_sobrescrever([saida], a.forcar)
    n = gold.exportar_revisao(ler_jsonl(cand), indexacao.carregar_trechos(trechos_arq), saida)
    print(
        f"{n} linhas em {saida}. Preencha 'decisao' (aceitar, editar, descartar) e, ao editar, as colunas "
        "corrigidas; depois rode `qa-manual importar-revisao --csv` com o arquivo."
    )


def cmd_importar_revisao(a: argparse.Namespace) -> None:
    cfg = _cfg(a)
    csv_arq = _pedir_arquivo(a.csv, "CSV revisado", (".csv",))
    itens = gold.importar_revisao(csv_arq, cfg.abstencao.frase)
    revisado = _dir_dados(cfg) / "gold_revisado.jsonl"
    escrever_jsonl(revisado, itens)
    print(f"{len(itens)} itens revisados em {revisado}")
    gold_test = None
    if a.dividir:
        _nao_sobrescrever([cfg.paths.gold_dev, cfg.paths.gold_test], a.forcar)
        dev, test = gold.dividir(itens, cfg.gold.frac_dev, cfg.gold.seed)
        escrever_jsonl(cfg.paths.gold_dev, dev)
        escrever_jsonl(cfg.paths.gold_test, test)
        gold_test = test
        print(f"dev: {len(dev)} itens em {cfg.paths.gold_dev}; test: {len(test)} itens em {cfg.paths.gold_test}")
    if a.reservar_ppl:
        if gold_test is None and Path(cfg.paths.gold_test).is_file():
            gold_test = ler_jsonl(cfg.paths.gold_test)
        trechos = indexacao.carregar_trechos(cfg.paths.trechos)
        gold.reservar_ppl(trechos, gold_test, a.reservar_ppl, cfg.gold.seed, Path(cfg.paths.ppl_holdout))
        print(f"Trechos para perplexidade: {cfg.paths.ppl_holdout}")


def cmd_perplexidade(a: argparse.Namespace) -> None:
    from .perplexity import gguf_do_ollama, rodar_llama_perplexity

    cfg = _cfg(a)
    binario = _pedir_arquivo(a.binario, "Binário llama-perplexity")
    gguf_base = (
        gguf_do_ollama(cfg.modelos.leitor_base)
        if a.gguf_base == "ollama"
        else _pedir_arquivo(a.gguf_base, "GGUF do leitor base (ou 'ollama')")
    )
    gguf_aj = _pedir_arquivo(a.gguf_ajustado, "GGUF do leitor ajustado", (".gguf",))
    texto = _pedir_arquivo(a.texto, "Texto reservado (.txt)", (".txt",), cfg.paths.ppl_holdout)
    ctx = a.ctx or cfg.avaliacao.ppl_ctx
    ppl = {
        "base": rodar_llama_perplexity(binario, gguf_base, texto, ctx),
        "ajustado": rodar_llama_perplexity(binario, gguf_aj, texto, ctx),
    }
    run_id = novo_run_id("perplexidade")
    run_dir = criar_run_dir(cfg.paths.runs_dir, run_id)
    escrever_json(
        run_dir / "perplexidade.json",
        {
            "run_id": run_id,
            "criado_em": agora_iso(),
            "ctx": ctx,
            "perplexidade": ppl,
            "texto": str(texto),
            "texto_sha256": sha256_arquivo(texto),
            "gguf": {"base": str(gguf_base), "ajustado": str(gguf_aj)},
            "modelos": {"base": cfg.modelos.leitor_base, "ajustado": cfg.modelos.leitor_ajustado},
        },
    )
    print(f"Perplexidade (ctx {ctx}): base {ppl['base']:.3f} | ajustado {ppl['ajustado']:.3f}")
    print(f"Gravado: {run_dir / 'perplexidade.json'}")


def cmd_avaliar(a: argparse.Namespace) -> None:
    from .evaluate import avaliar, tabela_texto

    base = _cfg(a)
    gold_arq = _pedir_arquivo(a.gold, "Arquivo gold (.jsonl)", (".jsonl",))
    nomes = a.configs or base.avaliacao.configs
    leitores = a.leitores or base.avaliacao.leitores
    configs = [config.carregar_experimento(n, a.config, a.sobrescrever) for n in nomes]
    client = criar_cliente(base)
    run_dir, df = avaliar(
        gold_arq,
        configs,
        leitores,
        a.repeticoes or base.avaliacao.repeticoes,
        client,
        retomar=a.retomar,
        perplexidade_path=Path(a.perplexidade) if a.perplexidade else None,
        limite=a.limite,
    )
    print("\n" + tabela_texto(df))
    pareado = run_dir / "pareado.csv"
    if pareado.is_file() and pareado.stat().st_size > 1:
        import pandas as pd

        try:
            dp = pd.read_csv(pareado)
        except pd.errors.EmptyDataError:
            dp = pd.DataFrame()
        if not dp.empty:
            print("\nGanho pareado (ajustado - base), IC 95% por bootstrap:\n" + tabela_texto(dp))
    print(f"\nSaídas em {run_dir}/: respostas.jsonl, metrics.csv, pareado.csv, gráficos PNG, log.txt")


def cmd_smoke(a: argparse.Namespace) -> int:
    import importlib.metadata
    import platform

    falhas = 0

    def item(ok: bool, texto: str) -> None:
        nonlocal falhas
        falhas += not ok
        print(f"[{'ok' if ok else 'falha'}] {texto}")

    cfg = _cfg(a)
    print(f"qa-manual {__version__} | Python {platform.python_version()} | {platform.platform()}")
    # Só lê as versões instaladas: importar torch e faiss no mesmo processo aborta no macOS (OpenMP duplicado).
    for pacote in (
        "pymupdf",
        "bm25s",
        "faiss-cpu",
        "numpy",
        "ollama",
        "nltk",
        "pandas",
        "matplotlib",
        "torch",
        "transformers",
        "bert-score",
    ):
        try:
            item(True, f"{pacote} {importlib.metadata.version(pacote)}")
        except importlib.metadata.PackageNotFoundError:
            item(False, f"{pacote} não instalado; rode `pip install -r requirements.txt`")
    try:
        from .tokenize_pt import stopwords_pt

        item(True, f"stop-words do NLTK ({len(stopwords_pt())} palavras)")
    except ErroUsuario as e:
        item(False, str(e))

    import faiss
    import numpy as np

    idx = faiss.IndexFlatIP(4)
    idx.add(np.eye(4, dtype=np.float32))
    item(int(idx.search(np.eye(4, dtype=np.float32)[:1], 1)[1][0][0]) == 0, "FAISS IndexFlatIP")

    import bm25s

    bm = bm25s.BM25()
    bm.index([["porto", "carga"], ["navio", "atracação"]], show_progress=False)
    docs, _ = bm.retrieve([["navio"]], k=1, show_progress=False)
    item(int(docs[0][0]) == 1, "bm25s")

    try:
        client = criar_cliente(cfg)
        disponiveis = client.modelos_disponiveis()
        item(True, f"Ollama {client.versao()} em {client.host}")
        for m in (cfg.modelos.leitor_base, cfg.modelos.embeddings, cfg.modelos.gerador, cfg.modelos.leitor_ajustado):
            opcional = m == cfg.modelos.leitor_ajustado
            presente = modelo_presente(m, disponiveis)
            if presente or not opcional:
                item(presente, f"modelo {m}" + ("" if presente else f": {instrucao_modelo(m)}"))
            else:
                print(f"[aviso] modelo {m} ainda não criado (passo 7): {instrucao_modelo(m)}")
        if modelo_presente(cfg.modelos.leitor_base, disponiveis):
            t0 = time.perf_counter()
            r = client.chat(
                model=cfg.modelos.leitor_base,
                messages=[{"role": "user", "content": "Responda só: ok"}],
                options=cfg.decodificacao.opcoes() | {"num_predict": 8},
                think=False,
            )
            item(bool(r["content"].strip()), f"chat com {cfg.modelos.leitor_base} ({time.perf_counter() - t0:.1f} s)")
        if modelo_presente(cfg.modelos.embeddings, disponiveis):
            v = client.embed(cfg.modelos.embeddings, ["teste de embedding"])
            item(len(v[0]) == cfg.denso.dimensao, f"embed com {cfg.modelos.embeddings} (dimensão {len(v[0])})")
    except ErroUsuario as e:
        item(False, str(e))

    if not a.sem_bertscore:
        try:
            from .metrics import bertscore_f1

            t0 = time.perf_counter()
            f = bertscore_f1(
                ["o navio atracou"],
                ["o navio atracou"],
                cfg.avaliacao.bertscore_modelo,
                cfg.avaliacao.bertscore_num_layers,
            )
            item(f[0] > 0.9, f"BERTScore em CPU ({cfg.avaliacao.bertscore_modelo}, {time.perf_counter() - t0:.1f} s)")
        except Exception as e:  # noqa: BLE001 - qualquer falha do bert-score é reportada
            item(False, f"BERTScore: {type(e).__name__}: {e}")

    servico = pasta_sincronizada(Path(cfg.paths.trechos).parent)
    if servico:
        print(
            f"[aviso] {Path(cfg.paths.trechos).parent.resolve()} está em pasta sincronizada ({servico}); "
            "dados do manual sairiam da máquina. Mova o projeto para uma pasta local."
        )
    if a.sem_rede:
        try:
            socket.create_connection(("pypi.org", 443), timeout=3).close()
            item(False, "rede externa acessível (esperado: sem rede)")
        except OSError:
            item(True, "rede externa inacessível, como esperado")
    print("\nTudo certo." if not falhas else f"\n{falhas} verificação(ões) falharam.")
    return 0 if not falhas else 1


def cmd_stats(a: argparse.Namespace) -> None:
    import json

    cfg = _cfg(a)
    for rotulo, caminho in (
        ("trechos_stats.json", indexacao.caminho_stats(cfg)),
        ("manifest.json", indexacao.caminho_manifesto(cfg)),
    ):
        if not Path(caminho).is_file():
            raise ErroUsuario(f"{caminho} não encontrado; rode `qa-manual indexar --manual <pdf>`")
        print(f"== {rotulo} ({caminho})\n{json.dumps(ler_json(caminho), ensure_ascii=False, indent=2)}\n")
    n_temas = ler_json(indexacao.caminho_stats(cfg)).get("n_temas", 0)
    if n_temas < indexacao.MIN_TEMAS:
        print(f"aviso: só {n_temas} tema(s); confira ingestao.regex_titulo para o estilo de títulos do manual")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def construir_parser() -> argparse.ArgumentParser:
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--config", default=config.BASE_PADRAO, help="YAML base (padrão: configs/base.yaml)")
    comum.add_argument("--sobrescrever", help="YAML aplicado sobre o base (ex.: configs/S3.yaml)")
    comum.add_argument("-v", "--verbose", action="store_true", help="log detalhado")

    p = argparse.ArgumentParser(prog="qa-manual", description="QA sobre manual técnico com LLM local (Ollama).")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="comando", required=True, metavar="COMANDO")

    s = sub.add_parser("indexar", parents=[comum], help="passos 1-4: PDF -> trechos -> índices BM25 e denso")
    s.add_argument("--manual", help="manual em PDF")
    s.add_argument("--reservar-ppl", type=int, metavar="N", help="reserva N trechos para a perplexidade")
    s.add_argument("--max-tokens", type=int, help="sobrescreve trechos.max_tokens")
    s.add_argument("--sobreposicao-paragrafos", type=int, help="sobrescreve trechos.sobreposicao_paragrafos")
    s.add_argument(
        "--permitir-pasta-sincronizada",
        action="store_true",
        help="aceita gravar em pasta do iCloud/Dropbox/OneDrive (só para documentos públicos de teste)",
    )
    s.set_defaults(fn=cmd_indexar)

    s = sub.add_parser("recuperar", parents=[comum], help="passos 8-9: depuração da recuperação")
    s.add_argument("--pergunta")
    s.add_argument("--modo", choices=("bm25", "denso", "hibrido"), default="hibrido")
    s.add_argument("--k", type=int)
    s.set_defaults(fn=cmd_recuperar)

    s = sub.add_parser("perguntar", parents=[comum], help="passos 8-11: responde perguntas")
    s.add_argument("--config-exp", default="S3", help="S0, S1, S2 ou S3 (padrão: S3)")
    s.add_argument("--leitor", choices=config.LEITORES, default="base")
    grupo = s.add_mutually_exclusive_group()
    grupo.add_argument("--pergunta")
    grupo.add_argument("--arquivo", help="arquivo .txt com uma pergunta por linha")
    s.set_defaults(fn=cmd_perguntar)

    s = sub.add_parser("gerar-perguntas", parents=[comum], help="passo 12: gera candidatas com o LLM local")
    s.add_argument("--n", type=int, help="número de candidatas (padrão: gold.n_candidatas)")
    s.add_argument("--trechos", help="trechos.jsonl (padrão: paths.trechos)")
    s.add_argument("--saida", help="padrão: data/gold_candidatas.jsonl")
    s.add_argument("--forcar", action="store_true", help="sobrescreve a saída")
    s.set_defaults(fn=cmd_gerar_perguntas)

    s = sub.add_parser("exportar-revisao", parents=[comum], help="passo 12: CSV para revisão humana")
    s.add_argument("--candidatas", help="padrão: data/gold_candidatas.jsonl")
    s.add_argument("--trechos", help="trechos.jsonl (padrão: paths.trechos)")
    s.add_argument("--saida", help="padrão: data/gold_revisao.csv")
    s.add_argument("--forcar", action="store_true", help="sobrescreve o CSV (apaga a revisão existente)")
    s.set_defaults(fn=cmd_exportar_revisao)

    s = sub.add_parser("importar-revisao", parents=[comum], help="passo 12: aplica a revisão e divide dev/teste")
    s.add_argument("--csv", help="CSV revisado")
    s.add_argument("--dividir", action="store_true", help="grava gold_dev.jsonl e gold_test.jsonl (split por tema)")
    s.add_argument("--reservar-ppl", type=int, metavar="N", help="reserva N trechos de temas fora do teste")
    s.add_argument("--forcar", action="store_true", help="sobrescreve gold_dev/gold_test existentes")
    s.set_defaults(fn=cmd_importar_revisao)

    s = sub.add_parser("perplexidade", parents=[comum], help="passo 14: perplexidade com llama-perplexity")
    s.add_argument("--binario", help="caminho do llama-perplexity")
    s.add_argument("--gguf-base", help="GGUF do leitor base, ou 'ollama' para usar o blob do Ollama")
    s.add_argument("--gguf-ajustado", help="GGUF do leitor ajustado (models/qwen3-manual-q4_k_m.gguf)")
    s.add_argument("--texto", help="padrão: paths.ppl_holdout")
    s.add_argument("--ctx", type=int, help="padrão: avaliacao.ppl_ctx")
    s.set_defaults(fn=cmd_perplexidade)

    s = sub.add_parser("avaliar", parents=[comum], help="passos 13 e 15: avaliação completa")
    s.add_argument("--gold", help="gold_dev.jsonl ou gold_test.jsonl")
    s.add_argument("--configs", nargs="+", help="padrão: avaliacao.configs")
    s.add_argument("--leitores", nargs="+", choices=config.LEITORES, help="padrão: avaliacao.leitores")
    s.add_argument("--repeticoes", type=int)
    s.add_argument("--retomar", metavar="RUN_ID", help="continua uma execução interrompida")
    s.add_argument("--perplexidade", help="perplexidade.json (padrão: o mais recente em runs/)")
    s.add_argument("--limite", type=int, help="avalia só as N primeiras perguntas")
    s.set_defaults(fn=cmd_avaliar)

    s = sub.add_parser("smoke", parents=[comum], help="testa Ollama, FAISS, bm25s e bert-score; imprime versões")
    s.add_argument("--sem-rede", action="store_true", help="confirma que a rede externa está desligada")
    s.add_argument("--sem-bertscore", action="store_true", help="pula o teste do bert-score")
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("stats", parents=[comum], help="mostra trechos_stats.json e manifest.json")
    s.set_defaults(fn=cmd_stats)
    return p


def _configurar_log(verbose: bool) -> logging.Handler:
    logger = logging.getLogger("qa_manual")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    h.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.addHandler(h)
    return h


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada; devolve o código de saída."""
    a = construir_parser().parse_args(argv)
    handler = _configurar_log(a.verbose)
    try:
        codigo = a.fn(a)
        return codigo if isinstance(codigo, int) else 0
    except ErroUsuario as e:
        print(f"erro: {e}", file=sys.stderr)
        return e.codigo
    except KeyboardInterrupt:
        dica = " Retome com --retomar <run_id> (veja runs/)." if a.comando == "avaliar" else ""
        print(f"\ninterrompido.{dica}", file=sys.stderr)
        return 130
    finally:
        logging.getLogger("qa_manual").removeHandler(handler)


if __name__ == "__main__":
    sys.exit(main())
