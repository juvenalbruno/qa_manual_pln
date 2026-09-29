# Question answering sobre manual técnico com LLM local

IC0024, PGCOMP/UFBA. Autores: _preencher_.

> Modelo de relatório. Os comentários `<!-- fonte: ... -->` indicam de qual arquivo vem cada número.
> Exemplos de perguntas e respostas devem ser **parafraseados**: o manual é confidencial.

## 1. Formulação da tarefa

QA em domínio fechado, open-book e generativo: dada uma pergunta em português, o sistema produz uma resposta
curta citando seção e página, ou "Não encontrado no manual". Restrições: modelos abertos servidos pelo Ollama,
execução local em CPU (16 GB de RAM como cenário base) e nenhum dado enviado a serviços externos.

Perguntas de pesquisa:
1. Quanto a recuperação melhora o leitor em relação ao cenário closed-book (S0 × S1-S3)?
2. Qual representação recupera melhor neste domínio: léxica, densa ou híbrida?
3. O sistema se abstém corretamente quando a informação não está no manual?

## 2. Corpus e pré-processamento

<!-- fonte: index/stats.json -->
| Estatística | Valor |
|---|---|
| Páginas | |
| Palavras | |
| Seções detectadas | |
| Passagens | |
| Palavras por passagem (mediana / p90 / máx.) | |
| Tabelas convertidas para Markdown | |
| Linhas de cabeçalho/rodapé removidas | |

Descrever a extração com PyMuPDF, a heurística de títulos, a limpeza (cabeçalhos repetidos, hifenização,
ligaduras, tabelas que continuam na página seguinte) e a segmentação: janelas de ~300 palavras que respeitam as
seções, com ~37 palavras de sobreposição. Relatar o resultado da inspeção de 20 passagens.

## 3. Conjunto de avaliação

<!-- fonte: saída de dividir-gold; data/revisoes/concordancia.json -->
| | Dev | Teste |
|---|---|---|
| factual | | |
| procedimental | | |
| sem_resposta | | |
| Seções | | |

Geração com `llama3.1:8b` (distinto do leitor), amostragem estratificada por seção, revisão manual de 100% das
candidatas, concordância entre anotadores (kappa) e divisão por seção sem sobreposição.

## 4. Representação e recuperação

BM25 (`bm25s`, tokens em minúsculas e sem acentos, sem stopwords); denso (`bge-m3`, vetores normalizados, FAISS
`IndexFlatIP`); híbrido por RRF (`Σ 1/(60 + posição)`). O título da seção é prefixado ao texto indexado.

<!-- fonte: runs/<ts>/recall_k.png e recall_k.csv -->
![Recall@k por modo](../runs/TIMESTAMP/recall_k.png)

## 5. Leitura e geração

Leitor `qwen3:8b` com `think: false`, temperatura 0, seed 42, `num_ctx` 8192 e no máximo 200 tokens. Prompt em
`prompts/leitor.txt`. A abstenção é detectada pela frase "Não encontrado no manual".

## 6. Configurações

| Config | Recuperação | Passagens no prompt |
|---|---|---|
| S0 | nenhuma (closed-book) | 0 |
| S1 | BM25 | k |
| S2 | denso | k |
| S3 | híbrido RRF | k |

Parâmetros escolhidos no dev pela varredura (tamanho × k): <!-- fonte: runs/<ts>_varredura/varredura.json -->

## 7. Avaliação

<!-- fonte: runs/<ts>/analise.md (média ± desvio em 3 repetições) -->
| Config | Recall@k | MRR | EM | F1 | Juiz | Fidelidade | Abst. correta | Abst. indevida | Lat. mediana |
|---|---|---|---|---|---|---|---|---|---|
| S0 | | | | | | | | | |
| S1 | | | | | | | | | |
| S2 | | | | | | | | | |
| S3 | | | | | | | | | |

Comparações pareadas (Δ F1 com IC 95% por bootstrap; McNemar): <!-- fonte: comparacoes.json -->

Validação do juiz em 50 itens: concordância exata ___, kappa ponderado ___. <!-- fonte: metrics.json/_validacao_juiz -->

## 8. Análise

<!-- fonte: runs/<ts>/erros_classificados.jsonl (tabela em analise.md) -->
| Categoria de erro | n | % |
|---|---|---|
| Falha de recuperação | | |
| Falha de leitura | | |
| Abstenção indevida | | |
| Alucinação | | |
| Erro do gold | | |

Discutir dois exemplos parafraseados (um acerto e um erro) e o efeito do tamanho do leitor (opcional,
`S3_gemma3_4b`).

## 9. Limitações

Gold pequeno e produzido pelo grupo (relatar ICs); juiz automático com possível viés (mitigado com modelo
distinto e validação manual); heurística de títulos dependente do layout; latência em CPU; um único manual.
