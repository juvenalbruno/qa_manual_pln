"""Processo isolado para o BERTScore (ver :func:`qa_manual.metrics.bertscore_f1`).

Lê ``{"preds", "refs", "modelo", "num_layers"}`` (JSON) da entrada padrão e escreve a lista de F1 na saída padrão.
"""

from __future__ import annotations

import importlib
import json
import sys

MAX_TOKENS_BERT = 512


def _limitar_tokenizador() -> None:
    """O tokenizador do BERTimbau declara ``model_max_length`` ~1e30; o bert-score repassa esse valor como limite
    de truncamento e o ``tokenizers`` estoura (``OverflowError ... max_length``). Limita a 512, o máximo do BERT."""
    modulo = importlib.import_module("bert_score.score")
    original = modulo.get_tokenizer

    def get_tokenizer(*args, **kwargs):
        tok = original(*args, **kwargs)
        if tok.model_max_length > 100_000:
            tok.model_max_length = MAX_TOKENS_BERT
        return tok

    modulo.get_tokenizer = get_tokenizer


def main() -> None:
    dados = json.load(sys.stdin)
    _limitar_tokenizador()
    from bert_score import score

    _, _, f1 = score(
        dados["preds"],
        dados["refs"],
        model_type=dados["modelo"],
        num_layers=dados["num_layers"],
        lang="pt",
        rescale_with_baseline=False,
        device="cpu",
        verbose=False,
    )
    json.dump([float(x) for x in f1], sys.stdout)


if __name__ == "__main__":
    main()
