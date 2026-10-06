---
marp: true
paginate: true
title: QA sobre manual técnico com RAG local
---

# QA sobre manual técnico com RAG local

IC0024 · PGCOMP/UFBA
_Autores_

> Modelo de slides em formato [Marp](https://marp.app), coerente com `docs/relatorio.md`. Os comentários
> `<!-- fonte: ... -->` indicam de onde vem cada número e viram notas do apresentador no Marp. Exemplos sempre
> parafraseados.

---

## Tarefa e restrições

- Pergunta em português → resposta curta + "(seção X, p. N)", ou "Não encontrado no manual"
- Domínio fechado, open-book, generativo: um manual técnico confidencial do setor portuário
- Modelos abertos no Ollama, tudo local em CPU
- Nenhum dado do manual sai da máquina; o Colab só vê um documento **público**

---

## Perguntas de pesquisa

1. Quanto a recuperação melhora o leitor em relação a S0?
2. Que recuperador funciona melhor: BM25, denso ou híbrido?
3. O ajuste fino QLoRA com dados públicos do setor ajuda no manual?
4. O sistema se abstém quando a resposta não está no manual?

---

## Arquitetura

```
Manual PDF ─► indexar ─► data/trechos.jsonl (t0001…) ─┬─► BM25 (bm25s)
                                                      └─► bge-m3 ─► FAISS IndexFlatIP
Pergunta ─► top-5: S0 nenhum | S1 BM25 | S2 denso | S3 RRF (c=60, 20 por lista)
         ─► leitor qwen3:4b | qwen3-manual:4b (temp. 0, seed 42, think false)
         ─► resposta + "(seção X, p. N)"  ou  "Não encontrado no manual"

Regulamento público ─► pares (llama3.2:3b) ─► QLoRA no Colab (T4) ─► GGUF q4_k_m ─► qwen3-manual:4b
```

---

## Pré-processamento

<!-- fonte: data/trechos_stats.json -->
- PyMuPDF por blocos; cabeçalhos/rodapés em > 50% das páginas removidos (___ blocos)
- Hifenização desfeita; títulos numerados por regex com validação de sequência
- Trechos de até ~400 tokens (1,3 token/palavra), mesmo tema, sem cortar parágrafo
- ___ páginas → ___ trechos em ___ seções; mediana de ___ tokens por trecho

---

## Configurações

| Config | Recuperação | Leitores |
|---|---|---|
| S0 | nenhuma (closed-book) | base · ajustado |
| S1 | BM25 | base · ajustado |
| S2 | denso (bge-m3 + FAISS) | base · ajustado |
| S3 | híbrido RRF | base · ajustado |

Fixos: k = 5, temperature 0, seed 42, `num_ctx` 8192, `num_predict` 200. Só recuperador e leitor mudam.

---

## Ajuste fino (QLoRA)

<!-- fonte: notebook do Colab; colab/treino_log.csv -->
- Qwen3-4B com Unsloth no Colab (T4): r = 16, alpha = 16, 2 épocas, lr 2e-4
- ___ pares gerados pelo `llama3.2:3b` a partir de um regulamento portuário **público**
- Loss de treino: ___ → ___; validação: ___ → ___
- Exportado em GGUF q4_k_m e registrado no Ollama como `qwen3-manual:4b`
- O modelo nunca viu o manual

---

## Conjunto de avaliação

<!-- fonte: data/gold_revisao.csv; data/gold_dev.jsonl; data/gold_test.jsonl -->
- 150 candidatas do `llama3.2:3b`: 20% sem resposta, respondíveis 60/40 factual/procedimental
- 100% revisadas no CSV: ___ aceitas, ___ editadas, ___ descartadas
- Dev ___ / teste ___ itens, divididos por seção (30/70), sem vazamento
- Teste: ___ factuais, ___ procedimentais, ___ sem resposta

---

## Resultados: resposta e recuperação (teste)

<!-- fonte: runs/<run_id>/metrics.csv -->
| Config | Leitor | EM | F1 | BERTScore | Recall@5 | MRR@5 |
|---|---|---|---|---|---|---|
| S0 | base / ajustado | | | | – | – |
| S1 | base / ajustado | | | | | |
| S2 | base / ajustado | | | | | |
| S3 | base / ajustado | | | | | |

---

## EM, F1 e BERTScore por configuração

<!-- fonte: runs/<run_id>/respostas_por_config.png -->
![w:1000](../runs/RUN_ID/respostas_por_config.png)

---

## Recall@5 por configuração

<!-- fonte: runs/<run_id>/recall_por_config.png -->
![w:700](../runs/RUN_ID/recall_por_config.png)

---

## Abstenção, citação e latência

<!-- fonte: runs/<run_id>/metrics.csv -->
| Config | Leitor | Abst. correta | Abst. indevida | Fonte válida | Lat. mediana / p95 (s) |
|---|---|---|---|---|---|
| S0 | base / ajustado | | | – | |
| S1 | base / ajustado | | | | |
| S2 | base / ajustado | | | | |
| S3 | base / ajustado | | | | |

---

## O ajuste fino ajudou?

<!-- fonte: runs/<run_id>/pareado.csv; runs/<run_ppl>/perplexidade.json -->
| Config | Δ EM [IC 95%] | Δ F1 [IC 95%] |
|---|---|---|
| S0 | | |
| S1 | | |
| S2 | | |
| S3 | | |

Perplexidade nos 30 trechos reservados: base ___ · ajustado ___

---

## Erros (___ classificados, config ___)

<!-- fonte: classificação manual sobre runs/<run_id>/respostas.jsonl e data/gold_test.jsonl -->
| Categoria | % |
|---|---|
| Erro do gold | |
| Falha de recuperação | |
| Abstenção indevida | |
| Falha de leitura | |
| Alucinação em sem resposta | |
| Correta, penalizada pela métrica | |

---

## Exemplos (parafraseados)

**Acerto:** _pergunta_ → _resposta_ (seção X, p. N), config ___

**Erro:** _pergunta_ → _resposta_; causa: _categoria_

---

## Conclusões

- PP1, recuperação × S0: ___
- PP2, melhor recuperador: ___
- PP3, ajuste fino: ___
- PP4, abstenção: ___

---

## Limitações e ameaças à validade

- Gold pequeno, gerado por LLM e revisado pelo grupo; perguntas de um único trecho favorecem o BM25
- EM é rígido para respostas generativas; BERTScore sem reescala; a fonte válida não prova o suporte da resposta
- Um manual, um domínio, leitores de 4B quantizados, CPU
- A perplexidade mede adaptação ao domínio, não qualidade de QA

---

## Referências principais

- Cortes, Vieira e Barone (2024), cap. 16 do BPLN, 2ª ed.
- Lewis et al. (2020), RAG · Karpukhin et al. (2020), DPR
- Robertson e Zaragoza (2009), BM25 · Cormack et al. (2009), RRF
- Zhang et al. (2020), BERTScore · Dettmers et al. (2023), QLoRA
- Zhang et al. (2023), ACL · Zaib et al. (2022), KAIS · Perera et al. (2025), arXiv:2509.05716
