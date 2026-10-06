#!/usr/bin/env bash
# Registra o leitor ajustado no Ollama local como qwen3-manual:4b (passo 7, parte local).
# Uso: bash colab/criar_modelo_local.sh models/qwen3-manual-q4_k_m.gguf
set -euo pipefail

GGUF="${1:-}"
if [ -z "$GGUF" ]; then
  read -r -p "GGUF do leitor ajustado: " GGUF
fi
if [ ! -f "$GGUF" ]; then
  echo "erro: GGUF não encontrado: $GGUF" >&2
  exit 2
fi
command -v ollama > /dev/null || { echo "erro: comando ollama não encontrado no PATH" >&2; exit 2; }

BASE="${BASE:-qwen3:4b}"
NOME="${NOME:-qwen3-manual:4b}"
CAMINHO="$(cd "$(dirname "$GGUF")" && pwd)/$(basename "$GGUF")"

# Mesmo TEMPLATE, PARAMETER e SYSTEM do modelo base; só o FROM aponta para o GGUF ajustado.
ollama show "$BASE" --modelfile | sed "s|^FROM .*|FROM $CAMINHO|" > Modelfile
ollama create "$NOME" -f Modelfile
echo "Teste rápido:"
ollama run "$NOME" "Responda apenas: ok"
