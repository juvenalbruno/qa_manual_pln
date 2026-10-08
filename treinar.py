#!/usr/bin/env python3
"""Ajuste fino (LoRA) do leitor com o texto do manual (arquivo separado do fluxo principal).

O leitor (Qwen3-4B Instruct) é treinado mais um pouco sobre o próprio texto do manual, para ficar mais familiarizado
com o vocabulário e o conteúdo dele. O resultado vira um novo modelo no Ollama, ``qwen3-manual``, que o ``avaliar.py``
compara automaticamente com o leitor original. Tudo roda na máquina (Mac com Apple Silicon, via MLX).

Passos:
    1. Dados ......... todos os trechos do índice (o manual inteiro) viram ``modelos/dados/train.jsonl``.
    2. Treino ........ LoRA (QLoRA em 4 bits) com o mlx-lm sobre a versão MLX do mesmo modelo do leitor.
    3. Adaptador ..... o adaptador treinado (poucos MB) é convertido para GGUF (conversor de LoRA do llama.cpp).
    4. Junção ........ o adaptador é somado ao GGUF que o Ollama já tem do leitor (llama-export-lora). Só as camadas
                       treinadas mudam (e saem em 16 bits); o resto é copiado como está.
    4b. Compactação .. as camadas treinadas voltam para o mesmo tipo de 4/6 bits do original (llama-quantize), e o
                       modelo ajustado fica do mesmo tamanho do original (~2,5 GB).
    5. Ollama ........ o resultado é registrado como ``qwen3-manual``, com o mesmo formato de conversa do leitor.

Espaço em disco: o pico fica em torno de 6 GB (no passo 4) e, no fim, sobram ~2,5 GB (o qwen3-manual no Ollama).
Os pesos MLX baixados para o treino (~2,3 GB) e os arquivos intermediários são apagados ao longo do caminho.

Uso:
    python treinar.py                 # treina e registra o qwen3-manual (uma vez; leva de minutos a horas)
    python treinar.py --epocas 1      # treino mais curto

Depois:
    python qa_manual.py perguntar --modelo qwen3-manual   # usar o modelo ajustado
    python avaliar.py                                     # compara o leitor original com o qwen3-manual

Downloads da primeira vez: os pesos MLX do leitor (~2,3 GB, Hugging Face). Nenhum texto do manual sai da máquina.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Usa o índice e a configuração do fluxo principal, que está na mesma pasta.
import avaliar as av
import qa_manual as qa

# Versão MLX (4 bits) do mesmo modelo do leitor (Qwen3-4B Instruct 2507); o treino parte dela. Precisa ser a mesma
# versão do leitor do Ollama, porque o adaptador treinado é somado aos pesos dele.
MODELO_BASE_MLX = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
# Leitor do Ollama que recebe o adaptador (e de onde vem o formato de conversa).
MODELO_BASE_OLLAMA = qa.MODELO_LEITOR
# Onde o README manda baixar e compilar o llama.cpp (conversor de LoRA e llama-export-lora).
LLAMA_CPP = Path("~/llama.cpp").expanduser()
CONVERSOR_LORA = LLAMA_CPP / "convert_lora_to_gguf.py"
EXPORT_LORA = LLAMA_CPP / "build" / "bin" / "llama-export-lora"
QUANTIZE = LLAMA_CPP / "build" / "bin" / "llama-quantize"

# Semente do treino (mesma ordem dos exemplos a cada execução).
SEMENTE = 42
# Hiperparâmetros do LoRA, pensados para caber em um Mac com 8 GB de memória.
EPOCAS = 2  # quantas vezes cada trecho de treino é visto
CAMADAS_LORA = 8  # quantas camadas (as últimas) recebem o adaptador
TAXA_APRENDIZADO = 1e-4
MAX_TOKENS = 1024  # trechos maiores são cortados

# Arquivos e pastas do treino (todos dentro de modelos/, que não vai para o git).
PASTA_DADOS = qa.PASTA_MODELOS / "dados"
PASTA_ADAPTADOR = qa.PASTA_MODELOS / "adaptador"
PASTA_PEFT = qa.PASTA_MODELOS / "adaptador_peft"
ARQ_ADAPTADOR_GGUF = qa.PASTA_MODELOS / "adaptador.gguf"
ARQ_JUNTO = qa.PASTA_MODELOS / "junto-f16.gguf"
ARQ_TIPOS = qa.PASTA_MODELOS / "tipos.txt"
ARQ_GGUF = qa.PASTA_MODELOS / f"{qa.MODELO_TREINADO}.gguf"
ARQ_MODELFILE = qa.PASTA_MODELOS / "Modelfile"


def rodar(comando: list[str], etapa: str) -> None:
    """Roda um comando externo mostrando a saída dele no terminal.

    Entradas:
        comando: programa e argumentos.
        etapa: nome da etapa, para a mensagem de erro.
    Erros:
        ErroUsuario se o comando terminar com erro.
    """
    if subprocess.run(comando, check=False).returncode != 0:
        raise qa.ErroUsuario(f"a etapa '{etapa}' falhou (veja a mensagem acima)")


def preparar_dados(indice: qa.Indice) -> int:
    """Passo 1: grava o manual inteiro, trecho por trecho, como conversas no formato do mlx-lm.

    Cada trecho vira uma conversa curta: o usuário pede o conteúdo da seção e o modelo responde com o trecho. O
    formato de conversa (em vez de texto corrido) mantém o modelo respondendo e encerrando a resposta como antes;
    treinado com texto corrido, ele passa a "continuar o documento" e entra em repetição.

    Entradas:
        indice: índice do manual.
    Saídas:
        número de trechos gravados em ``modelos/dados/train.jsonl`` (uma conversa por linha). Não há conjunto de
        validação: o treino usa o manual todo.
    """
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    # Apaga uma validação antiga, se houver, para o mlx-lm não usá-la.
    (PASTA_DADOS / "valid.jsonl").unlink(missing_ok=True)
    with open(PASTA_DADOS / "train.jsonl", "w", encoding="utf-8") as arquivo:
        for trecho in indice.trechos:
            # A pergunta cita seção e página, para o modelo associar cada assunto ao lugar dele no manual.
            pedido = f"O que diz o manual na seção {trecho['secao']} (p. {trecho['pagina_inicio']})?"
            conversa = [{"role": "user", "content": pedido}, {"role": "assistant", "content": trecho["texto"]}]
            arquivo.write(json.dumps({"messages": conversa}, ensure_ascii=False) + "\n")
    return len(indice.trechos)


def treinar(n_treino: int, epocas: int) -> None:
    """Passo 2: treina o adaptador LoRA com o mlx-lm.

    Entradas:
        n_treino: número de exemplos de treino.
        epocas: quantas vezes cada exemplo é visto.
    Saídas:
        nenhuma; grava o adaptador em ``modelos/adaptador/``.
    """
    # Um exemplo por passo (lote 1, para caber na memória), acumulando 4 passos por atualização dos pesos.
    iteracoes = max(1, math.ceil(n_treino * epocas))
    rodar(
        [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", MODELO_BASE_MLX,
            "--train",
            "--mask-prompt",  # aprende só a resposta (o trecho), não o pedido
            "--data", str(PASTA_DADOS),
            "--adapter-path", str(PASTA_ADAPTADOR),
            "--iters", str(iteracoes),
            "--batch-size", "1",
            "--grad-accumulation-steps", "4",
            "--num-layers", str(CAMADAS_LORA),
            "--learning-rate", str(TAXA_APRENDIZADO),
            "--max-seq-length", str(MAX_TOKENS),
            "--grad-checkpoint",
            "--seed", str(SEMENTE),
        ],
        "treino",
    )  # fmt: skip


def converter_adaptador() -> None:
    """Passo 3: converte o adaptador do formato do MLX para GGUF.

    O conversor do llama.cpp lê adaptadores no formato PEFT (Hugging Face). Os dois formatos guardam as mesmas
    matrizes A e B do LoRA, só com outros nomes e transpostas; aqui elas são renomeadas e transpostas.

    Saídas:
        nenhuma; grava ``modelos/adaptador.gguf`` e apaga do cache os pesos MLX baixados para o treino (~2,3 GB).
    """
    import numpy as np
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from safetensors.numpy import load_file, save_file

    # Posto e escala do LoRA usados no treino (no PEFT a escala vira lora_alpha = escala x posto).
    parametros = json.loads((PASTA_ADAPTADOR / "adapter_config.json").read_text(encoding="utf-8"))["lora_parameters"]
    # MLX: "model.layers.N.mlp.down_proj.lora_a" (entrada x posto).
    # PEFT: "base_model.model.model.layers.N.mlp.down_proj.lora_A.weight" (posto x entrada).
    pesos = {}
    for nome, matriz in load_file(PASTA_ADAPTADOR / "adapters.safetensors").items():
        camada, tipo = nome.rsplit(".", 1)
        pesos[f"base_model.model.{camada}.lora_{tipo[-1].upper()}.weight"] = np.ascontiguousarray(matriz.T)
    PASTA_PEFT.mkdir(parents=True, exist_ok=True)
    save_file(pesos, PASTA_PEFT / "adapter_model.safetensors")
    # O conversor precisa só da configuração do modelo base (arquivos .json, já no cache desde o treino).
    base = snapshot_download(MODELO_BASE_MLX, allow_patterns=["*.json"])
    configuracao = {
        "peft_type": "LORA",
        "base_model_name_or_path": base,
        "r": parametros["rank"],
        "lora_alpha": parametros["scale"] * parametros["rank"],
        "target_modules": sorted({nome.split(".")[-3] for nome in pesos}),
    }
    (PASTA_PEFT / "adapter_config.json").write_text(json.dumps(configuracao, indent=1), encoding="utf-8")
    comando = [sys.executable, str(CONVERSOR_LORA), str(PASTA_PEFT), "--base", base]
    rodar([*comando, "--outfile", str(ARQ_ADAPTADOR_GGUF), "--outtype", "f16"], "conversão do adaptador")
    # Os pesos MLX só serviam para o treino: apaga do cache do Hugging Face para liberar espaço.
    shutil.rmtree(Path(HF_HUB_CACHE) / ("models--" + MODELO_BASE_MLX.replace("/", "--")), ignore_errors=True)


def juntar_no_leitor() -> None:
    """Passo 4: soma o adaptador ao GGUF do leitor que o Ollama já tem e compacta o resultado.

    Saídas:
        nenhuma; grava ``modelos/qwen3-manual.gguf``, com cada matriz no mesmo tipo (4, 6 ou 16 bits) que ela tem no
        leitor original. Assim o modelo ajustado tem o mesmo tamanho do original e a comparação entre os dois é justa.
    """
    # Usa o próprio arquivo do Ollama como base: nada de baixar ou converter o modelo inteiro de novo.
    base = av.gguf_do_ollama(MODELO_BASE_OLLAMA)
    # Soma o adaptador: as matrizes treinadas saem em 16 bits; as outras são copiadas sem mudança.
    rodar([str(EXPORT_LORA), "-m", str(base), "--lora", str(ARQ_ADAPTADOR_GGUF), "-o", str(ARQ_JUNTO)], "junção")
    # Lista o tipo original de cada matriz (lendo só o cabeçalho do GGUF do Ollama, com a biblioteca do llama.cpp).
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    from gguf import GGUFReader

    tipos = [f"^{re.escape(t.name)}$={t.tensor_type.name}" for t in GGUFReader(str(base)).tensors]
    ARQ_TIPOS.write_text("\n".join(tipos) + "\n", encoding="utf-8")
    # Devolve cada matriz ao tipo original. O llama-quantize só mexe nas que mudaram de tipo (as treinadas).
    comando = [str(QUANTIZE), "--allow-requantize", "--tensor-type-file", str(ARQ_TIPOS)]
    rodar([*comando, str(ARQ_JUNTO), str(ARQ_GGUF), "Q4_K_M"], "compactação")
    ARQ_JUNTO.unlink()


def registrar_no_ollama() -> None:
    """Passo 5: cria o modelo ``qwen3-manual`` no Ollama a partir do GGUF junto.

    Copia o Modelfile do leitor (formato de conversa e parâmetros) trocando só o arquivo de pesos, para que o modelo
    ajustado converse do mesmo jeito que o original.

    Saídas:
        nenhuma; o modelo fica disponível no Ollama e o GGUF local é apagado (o Ollama guarda uma cópia).
    """
    saida = subprocess.run(
        ["ollama", "show", MODELO_BASE_OLLAMA, "--modelfile"], capture_output=True, text=True, check=False
    )
    if saida.returncode != 0:
        raise qa.ErroUsuario(f"não consegui ler o Modelfile de {MODELO_BASE_OLLAMA} no Ollama")
    # Troca a linha "FROM <pesos do leitor>" pelo GGUF ajustado.
    modelfile = re.sub(r"^FROM .*$", f"FROM {ARQ_GGUF}", saida.stdout, count=1, flags=re.MULTILINE)
    ARQ_MODELFILE.write_text(modelfile, encoding="utf-8")
    rodar(["ollama", "create", qa.MODELO_TREINADO, "-f", str(ARQ_MODELFILE)], "registro no Ollama")
    ARQ_GGUF.unlink()


def main(argv: list[str] | None = None) -> int:
    """Lê os argumentos e executa os cinco passos do treino.

    Saídas:
        código de saída: 0 sucesso, 2 erro, 130 interrompido.
    """
    parser = argparse.ArgumentParser(prog="treinar.py", description="Ajuste fino (LoRA) do leitor com o manual.")
    parser.add_argument("--epocas", type=float, default=EPOCAS, help=f"passadas pelos trechos (padrão: {EPOCAS})")
    args = parser.parse_args(argv)
    try:
        # O mlx-lm só existe para Mac com Apple Silicon.
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            raise qa.ErroUsuario("mlx-lm não instalado (só funciona em Mac com Apple Silicon)") from None
        # Confere o llama.cpp antes do treino, que é a parte demorada.
        for ferramenta in (CONVERSOR_LORA, EXPORT_LORA, QUANTIZE):
            if not ferramenta.is_file():
                raise qa.ErroUsuario(f"{ferramenta.name} não encontrado em {ferramenta.parent} (veja o README)")
        # O pico de uso do disco é de ~6 GB (no passo 4).
        if shutil.disk_usage(qa.RAIZ).free < 6 * 1024**3:
            print("aviso: menos de 8 GB livres no disco; a junção pode falhar", file=sys.stderr)
        inicio = time.perf_counter()
        indice = qa.carregar_indice()
        print("Passo 1/5: preparando os dados ...", file=sys.stderr)
        n_treino = preparar_dados(indice)
        print(f"Passo 2/5: treinando com os {n_treino} trechos do manual ...", file=sys.stderr)
        treinar(n_treino, args.epocas)
        print("Passo 3/5: convertendo o adaptador ...", file=sys.stderr)
        converter_adaptador()
        print("Passo 4/5: somando o adaptador ao leitor e compactando ...", file=sys.stderr)
        juntar_no_leitor()
        print(f"Passo 5/5: registrando {qa.MODELO_TREINADO} no Ollama ...", file=sys.stderr)
        registrar_no_ollama()
        minutos = (time.perf_counter() - inicio) / 60
        print(f"\nPronto em {minutos:.0f} min: o modelo {qa.MODELO_TREINADO} está no Ollama.")
        print("Para comparar com o original: python avaliar.py")
        return 0
    except qa.ErroUsuario as erro:
        print(f"erro: {erro}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrompido.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
