"""Processo isolado para o BERTScore: lê ``{"preds", "refs", "modelo", "num_layers"}`` (JSON) da entrada padrão e
escreve a lista de F1 na saída padrão. Ver :func:`qa_manual.metrics.bertscore_f1`."""

from __future__ import annotations

import json
import sys


def main() -> None:
    dados = json.load(sys.stdin)
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
