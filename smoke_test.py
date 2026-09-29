"""Etapa 0: verifica o ambiente e registra hardware e versões.

Uso:
    python smoke_test.py                  # testa leitor, embeddings e juiz no Ollama local
    python smoke_test.py --gravar-readme  # também atualiza a seção "Ambiente" do README.md

Critério de aceite: rodar com a rede desligada (Wi-Fi/cabo desconectados) e obter
uma resposta do leitor e um vetor do modelo de embeddings.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import requests

from src.ollama_client import OllamaClient
from src.utils import ErroUsuario, carregar_config_indexacao

RAIZ = Path(__file__).resolve().parent
BIBLIOTECAS = ("pymupdf", "bm25s", "faiss-cpu", "numpy", "requests", "pyyaml", "matplotlib")


def _cmd(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware() -> dict:
    sistema = platform.system()
    cpu, ram_gb, gpu = platform.processor() or platform.machine(), None, "nenhuma detectada"
    if sistema == "Darwin":
        cpu = _cmd("sysctl", "-n", "machdep.cpu.brand_string") or cpu
        mem = _cmd("sysctl", "-n", "hw.memsize")
        ram_gb = round(int(mem) / 2**30, 1) if mem.isdigit() else None
        if platform.machine() == "arm64":
            gpu = f"GPU integrada Apple Silicon (memória unificada, {ram_gb} GB)"
    elif sistema == "Linux":
        info = Path("/proc/cpuinfo").read_text(errors="ignore") if Path("/proc/cpuinfo").exists() else ""
        m = re.search(r"model name\s*:\s*(.+)", info)
        cpu = m.group(1).strip() if m else cpu
        mem = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
        m = re.search(r"MemTotal:\s*(\d+)", mem)
        ram_gb = round(int(m.group(1)) / 2**20, 1) if m else None
    if shutil.which("nvidia-smi"):
        gpu = _cmd("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader") or gpu
    return {
        "sistema": f"{sistema} {platform.release()} ({platform.machine()})",
        "cpu": cpu,
        "ram_gb": ram_gb,
        "gpu": gpu,
        "python": platform.python_version(),
    }


def versoes_bibliotecas() -> dict:
    saida = {}
    for lib in BIBLIOTECAS:
        try:
            saida[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            saida[lib] = "não instalada"
    return saida


def modelos_instalados(url: str) -> dict[str, dict]:
    dados = requests.get(f"{url}/api/tags", timeout=10).json()
    return {m["name"]: m for m in dados.get("models", [])}


def _resolver(nome: str, instalados: dict[str, dict]) -> str | None:
    if nome in instalados:
        return nome
    if ":" not in nome and f"{nome}:latest" in instalados:
        return f"{nome}:latest"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gravar-readme", action="store_true")
    args = ap.parse_args()

    cfg = carregar_config_indexacao()
    modelos = {"leitor": cfg["leitor"], "embeddings": cfg["embeddings"], "juiz": cfg["juiz"]}
    falhas = 0
    try:
        cliente = OllamaClient(cfg.get("ollama_url", "http://localhost:11434"))
        instalados = modelos_instalados(cliente.url)
    except requests.ConnectionError:
        print(f"[FALHA] Ollama não está respondendo em {cfg.get('ollama_url')}. "
              "Instale-o (https://ollama.com/download) e rode `ollama serve`.")
        return 1
    except (ErroUsuario, requests.RequestException) as e:
        print(f"[FALHA] Ollama: {e}")
        return 1
    print(f"[ OK ] Ollama {cliente.versao()} em {cliente.url}")

    tags = {}
    for papel, nome in modelos.items():
        real = _resolver(nome, instalados)
        if real is None:
            print(f"[FALHA] modelo {papel} '{nome}' não instalado. Rode: ollama pull {nome}")
            falhas += 1
            continue
        det = instalados[real].get("details", {})
        tags[papel] = {
            "tag": real,
            "digest": instalados[real].get("digest", "")[:12],
            "parametros": det.get("parameter_size"),
            "quantizacao": det.get("quantization_level"),
        }

    if "leitor" in tags:
        r = cliente.chat(modelos["leitor"], "Responda apenas com a palavra OK.", num_predict=10, think=False)
        ok = bool(r.texto)
        falhas += not ok
        print(f"[{' OK ' if ok else 'FALHA'}] /api/chat {modelos['leitor']}: {r.texto!r} ({r.latencia_s:.1f} s)")
    if "embeddings" in tags:
        v = cliente.embed(modelos["embeddings"], ["Qual o prazo para liberação da carga?"])[0]
        print(f"[ OK ] /api/embed {modelos['embeddings']}: vetor de dimensão {len(v)}")
    if "juiz" in tags:
        r = cliente.chat(modelos["juiz"], "Responda apenas com o número 2.", num_predict=5)
        print(f"[{' OK ' if r.texto else 'FALHA'}] /api/chat {modelos['juiz']}: {r.texto!r} ({r.latencia_s:.1f} s)")

    hw = hardware()
    libs = versoes_bibliotecas()
    bloco = ["| Item | Valor |", "|---|---|"]
    bloco += [f"| {k} | {v} |" for k, v in hw.items()]
    bloco.append(f"| Ollama | {cliente.versao()} |")
    for papel, t in tags.items():
        bloco.append(f"| Modelo ({papel}) | `{t['tag']}` ({t['parametros']}, {t['quantizacao']}, digest {t['digest']}) |")
    bloco += [f"| {lib} | {v} |" for lib, v in libs.items()]
    texto = "\n".join(bloco)
    print("\nAmbiente:\n" + texto)
    print("\nPara o critério de aceite, repita este teste com a rede desligada.")

    if args.gravar_readme:
        readme = RAIZ / "README.md"
        conteudo = readme.read_text(encoding="utf-8")
        novo = re.sub(
            r"(<!-- AMBIENTE:INICIO -->).*?(<!-- AMBIENTE:FIM -->)",
            lambda m: f"{m.group(1)}\n{texto}\n{m.group(2)}",
            conteudo,
            flags=re.DOTALL,
        )
        readme.write_text(novo, encoding="utf-8")
        print(f"Seção de ambiente atualizada em {readme}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
