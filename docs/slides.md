---
marp: true
paginate: true
title: QA sobre manual técnico com LLM local
---

# QA sobre manual técnico com LLM local

IC0024 · PGCOMP/UFBA
_Autores_

> Modelo de slides em formato [Marp](https://marp.app). Exemplos sempre parafraseados.

---

## Tarefa e restrições

- Pergunta em português → resposta curta + seção e página, ou "Não encontrado no manual"
- Domínio fechado, open-book, generativo
- Tudo local via Ollama: manual confidencial, CPU com 16 GB

---

## Arquitetura

```
Manual PDF ─► indexar ─► passages.jsonl ─┬─► BM25 (bm25s)
                                          └─► FAISS (bge-m3)
Pergunta ─► recuperar top-k (BM25 | denso | RRF) ─► qwen3:8b (temp. 0) ─► resposta + citação
gold_test.jsonl ─► avaliar ─► Recall@k, EM, F1, juiz llama3.1:8b, abstenção, latência
```

---

## Conjunto de avaliação

- ~100 perguntas: factual, procedimental, sem resposta (20%)
- Geradas por `llama3.1:8b`, 100% revisadas, 30% anotadas em dobro (kappa = ___)
- Dev 30% / teste 70%, divididos por seção

---

## Resultados (teste, 3 repetições)

| Config | Recall@5 | F1 | Juiz | Abst. correta | Latência |
|---|---|---|---|---|---|
| S0 | – | | | | |
| S1 BM25 | | | | | |
| S2 denso | | | | | |
| S3 híbrido | | | | | |

---

## Recall@k por modo

![w:800](../runs/TIMESTAMP/recall_k.png)

---

## Erros (50 classificados)

| Categoria | % |
|---|---|
| Recuperação | |
| Leitura | |
| Abstenção indevida | |
| Alucinação | |
| Erro do gold | |

---

## Exemplos (parafraseados)

**Acerto:** _pergunta_ → _resposta_ (seção X, p. N)

**Erro:** _pergunta_ → _resposta_; causa: _categoria_

---

## Conclusões e limitações

- _Resposta às perguntas de pesquisa_
- Gold pequeno, juiz automático validado em 50 itens, um único manual
