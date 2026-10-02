# Question answering sobre manual técnico com RAG local

IC0024, PGCOMP/UFBA. Autores: _preencher_. Data: _preencher_.

> Modelo de relatório. Os comentários `<!-- fonte: ... -->` indicam de qual arquivo vem cada número. `<run_id>` é a
> pasta da avaliação em `runs/` (por exemplo, `2026-10-10T18-02-11_avaliar`), e `<run_ppl>` é a pasta da medida de
> perplexidade (`..._perplexidade`). Campos `___` e células vazias são preenchidos com os resultados; nenhum
> número deve ser estimado. Exemplos de perguntas e respostas devem ser **parafraseados**: o manual é
> confidencial.

## Resumo

_Três a cinco frases: tarefa, abordagem (Retriever-Reader com leitor generativo, local em CPU), melhor
configuração, ganho da recuperação sobre S0, efeito do ajuste fino e taxa de abstenção correta._

## 1. Formulação da tarefa

QA em domínio fechado, open-book e generativo, sobre um único manual técnico confidencial do setor portuário,
em português. Dada uma pergunta, o sistema produz uma resposta curta, em uma ou duas frases, terminada pela fonte
no formato "(seção X, p. N)". Quando a informação não está nos trechos recuperados, a resposta é exatamente "Não
encontrado no manual".

A arquitetura é Retriever-Reader com leitor generativo, isto é, geração aumentada por recuperação (RAG; Lewis et
al., 2020): um recuperador seleciona k = 5 trechos do manual, e um LLM aberto, servido pelo Ollama, redige a
resposta a partir deles. Restrições:

- modelos abertos;
- execução local em CPU;
- nenhum dado do manual enviado a serviços externos.

### 1.1 Perguntas de pesquisa

1. **PP1.** Quanto a recuperação melhora o leitor em relação ao cenário sem recuperação (S0 × S1–S3)?
2. **PP2.** Qual recuperador funciona melhor neste manual: léxico (BM25), denso (bge-m3) ou híbrido (RRF)? Isso
   é medido no Recall@5 e no MRR@5, e no efeito sobre a resposta.
3. **PP3.** O ajuste fino QLoRA com dados **públicos** do setor portuário melhora o leitor no manual? Isso é medido
   no ganho pareado em EM e F1, na perplexidade, na citação e na abstenção.
4. **PP4.** O sistema se abstém quando a informação não está no manual? A que custo em abstenções indevidas?

### 1.2 Trabalhos relacionados

_Posicionar o trabalho em um ou dois parágrafos._ Sugestões:

- Cortes, Vieira e Barone (2024) apresentam sistemas de perguntas e respostas em português.
- Zhang et al. (2023) revisam a eficiência de sistemas Retriever-Reader em domínio aberto, o que é relevante para a
  execução em CPU.
- Zaib et al. (2022) e Perera et al. (2025) revisam QA conversacional. Este trabalho trata perguntas isoladas, de
  uma rodada, mas compartilha a decomposição em recuperação e geração e os desafios de avaliação.
- Os componentes vêm de:
  - BM25 (Robertson e Zaragoza, 2009);
  - recuperação densa por bi-encoder (Karpukhin et al., 2020);
  - fusão por RRF (Cormack et al., 2009);
  - RAG (Lewis et al., 2020);
  - BERTScore (Zhang et al., 2020);
  - QLoRA (Dettmers et al., 2023).

### 1.3 Ambiente

<!-- fonte: saída de `qa-manual smoke`; runs/<run_id>/config.json (versao_codigo) -->
| Item | Valor |
|---|---|
| CPU / RAM | |
| Sistema operacional | |
| Python / Ollama | |
| Versão do código | |
| Treino do ajuste fino | Google Colab, GPU T4 |

### 1.4 Confidencialidade

- O cliente do Ollama só aceita endereços locais. `indexar` recusa gravar o manual e seus derivados em pasta
  sincronizada com a nuvem.
- O Google Colab recebeu **apenas** o documento público e os pares de treino derivados dele (seção 4). Nenhum
  trecho, pergunta ou resposta do manual saiu da máquina.
- Os logs (`log.txt`) registram só ids, contagens, tempos e hashes. O cache do BERTScore guarda hashes e valores,
  nunca texto. O modelo BERTimbau é baixado do Hugging Face, o que é download e não envio de dados.
- `data/`, `index/`, `runs/`, `models/` e PDFs ficam fora do git. Os exemplos deste relatório são parafraseados, e
  as chaves de `trechos_por_tema` são só numerações de seção.

## 2. Corpus e pré-processamento

Manual: _tipo de documento, número de páginas, estrutura de seções; sem conteúdo identificável_.

A ingestão (`qa-manual indexar`) segue estes passos:

1. **Extração por blocos** com PyMuPDF (`page.get_text("blocks")`). Um PDF sem camada de texto é recusado (exigiria
   OCR local).
2. **Cabeçalhos e rodapés:** blocos que se repetem em mais de 50% das páginas (comparados sem dígitos) são
   removidos. Entradas de sumário ("Título ........ 17") são descartadas.
3. **Limpeza:**
   - hifenização de fim de linha desfeita ("libera-/ção" vira "liberação");
   - quebras simples trocadas por espaço;
   - ligaduras tipográficas desfeitas;
   - caracteres de controle removidos.
4. **Títulos numerados:**
   - detectados pela regex `^(\d+(\.\d+){0,3})\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{2,80}$`;
   - **validação de sequência:** uma numeração que não pode suceder a seção corrente (salto maior que 5, ou volta
     para trás) é tratada como texto. Isso evita que passos numerados e linhas de tabela virem seções.
   - o título define o tema (`tema_id`, por exemplo `4.2`) e não entra no texto dos trechos.
5. **Trechos:**
   - parágrafos consecutivos do mesmo tema são agrupados até cerca de 400 tokens, estimados como 1,3 token por
     palavra;
   - nenhum parágrafo é cortado: um parágrafo maior que 400 tokens fica sozinho;
   - grupos com menos de 60 tokens são fundidos com o seguinte do mesmo tema;
   - não há sobreposição;
   - parágrafos com menos de 25 caracteres são descartados;
   - os ids seguem a ordem do documento: `t0001`, `t0002`, ...

<!-- fonte: data/trechos_stats.json (chaves indicadas na tabela) -->
| Estatística | Chave em `trechos_stats.json` | Valor |
|---|---|---|
| Páginas | `n_paginas` | |
| Blocos extraídos | `n_blocos` | |
| Blocos de cabeçalho/rodapé removidos | `blocos_cabecalho_rodape_removidos` | |
| Títulos detectados | `n_titulos` | |
| Candidatos a título fora de sequência (tratados como texto) | `titulos_fora_de_sequencia` | |
| Parágrafos | `n_paragrafos` | |
| Parágrafos curtos descartados | `paragrafos_curtos_descartados` | |
| Trechos | `n_trechos` | |
| Temas (seções) | `n_temas` | |
| Tokens por trecho: média / mediana / mín. / máx. | `tokens` | |
| Trechos acima de 400 tokens (parágrafo longo isolado) | `acima_de_max_tokens` | |
| Trechos abaixo de 60 tokens (último do tema) | `abaixo_de_min_tokens` | |
| Trechos por tema: mín. / mediana / máx. | `trechos_por_tema` | |

Histograma de tokens por trecho (10 faixas): _tabela ou gráfico a partir de `histograma_tokens`_.
<!-- fonte: data/trechos_stats.json (histograma_tokens) -->

<!-- fonte: index/manifest.json -->
| Manifesto do índice | Valor |
|---|---|
| `hash_pdf_sha256` (12 primeiros caracteres) | |
| `hash_trechos_sha256` (12 primeiros caracteres) | |
| `modelo_embeddings` / `dimensao` / `normalizar_l2` | bge-m3 / 1024 / true |
| `bm25` (k1, b, stop-words, acentos) | 1,5 / 0,75 / removidas / mantidos |
| `versao_codigo` / `criado_em` | |

**Inspeção manual:** _resultado da leitura de 20 trechos sorteados de `data/trechos.jsonl`, por exemplo títulos
perdidos, tabelas embaralhadas, cabeçalhos remanescentes ou trechos que misturam assuntos. Relatar só contagens e
tipos de problema._

## 3. Configurações experimentais

Só o recuperador e o leitor mudam entre as execuções. Os demais parâmetros vêm de `configs/base.yaml`.

<!-- fonte: configs/S0.yaml a S3.yaml; runs/<run_id>/config.json (configs) -->
| Config | Recuperação | Trechos no prompt | Prompt |
|---|---|---|---|
| S0 | nenhuma (closed-book) | 0 | `prompts/leitor_s0.txt` |
| S1 | BM25 (`bm25s`) | 5 | `prompts/leitor.txt` |
| S2 | denso: bge-m3 + FAISS `IndexFlatIP` | 5 | `prompts/leitor.txt` |
| S3 | híbrido: RRF de BM25 e denso | 5 | `prompts/leitor.txt` |

Cada configuração roda com dois leitores, num total de 8 combinações:

- **base:** `qwen3:4b`;
- **ajustado:** `qwen3-manual:4b`, descrito na seção 4.

### 3.1 Recuperação

- **BM25** (Robertson e Zaragoza, 2009):
  - biblioteca `bm25s`, com k1 = 1,5 e b = 0,75;
  - tokens em minúsculas, com acentos mantidos e sem as stop-words do NLTK.
- **Denso:**
  - embeddings `bge-m3` (dimensão 1024) calculados pelo Ollama e normalizados (L2);
  - busca exata por produto interno no FAISS `IndexFlatIP`, equivalente ao cosseno;
  - recuperação por bi-encoder, como em Karpukhin et al. (2020).
- **Híbrido:**
  - Reciprocal Rank Fusion (Cormack et al., 2009) com score = Σ 1/(c + posição) e c = 60;
  - cada lista contribui com 20 candidatos antes da fusão;
  - empates são desfeitos pelo id.

O texto indexado é o do trecho, sem o título da seção. A abstenção por limiar de score está desligada
(`abstencao.limiar_score: null`): quem decide abster-se é o leitor.

### 3.2 Leitura e geração

- **Decodificação:**
  - temperature 0, top_p 1,0 e seed 42;
  - `num_ctx` 8192 e `num_predict` 200;
  - `think: false`; blocos `<think>` residuais são removidos.
- **Prompt:** cada trecho é apresentado como `[n] (seção {tema_id}, p. {pagina_inicio}) {texto}`. O prompt pede uma
  resposta em uma ou duas frases, terminada pela fonte, ou exatamente "Não encontrado no manual". Em S0 o prompt
  não tem trechos nem pede citação.
- **Abstenção:** detectada quando a resposta contém a frase de abstenção, ignorando maiúsculas, acentos e
  pontuação.
- **Citações:** extraídas por expressão regular.

**Uso do dev:** `data/gold_dev.jsonl` serviu para depurar prompts e pipeline. Os números reportados são do teste.
Ajustes feitos com base no dev: ___ (ou "nenhum").

## 4. Ajuste fino do leitor (QLoRA)

**Objetivo:** adaptar o `qwen3:4b` ao vocabulário do setor portuário e ao formato de resposta (resposta curta
seguida de "(seção X, p. N)") **sem que o modelo veja o manual**.

**Dados:**

- **Documento público:** _título, órgão emissor, ano, URL e licença do regulamento portuário_. Páginas: ___.
- **Pares de treino:** gerados no Colab pelo `llama3.2:3b` (Ollama) com `prompts/gerador_pares_treino.txt`, a
  partir de trechos do documento público com seção e página. Cada par tem uma pergunta factual ou procedimental e uma
  resposta de uma ou duas frases terminada pela fonte.
- **Formato de cada exemplo de treino:** o prompt de `prompts/leitor.txt` com o trecho-evidência e 2 a 4
  distratores (BM25) em ordem aleatória, até ~1400 tokens de trechos; alvo = resposta terminada pela fonte do
  trecho, ou `Não encontrado no manual.` em 20% dos exemplos. 5% dos pares vão para validação.

<!-- fonte: notebook do Colab (contagens impressas) -->
| Pares | Valor |
|---|---|
| Gerados / descartados (JSON inválido ou revisão) | |
| Treino / validação | |
| Factuais / procedimentais | |

<!-- fonte: notebook do Colab (configuração do treino) -->
| Hiperparâmetro | Valor |
|---|---|
| Modelo de partida (checkpoint Unsloth) | `unsloth/Qwen3-4B` |
| Quantização no treino | 4 bits (QLoRA, `load_in_4bit=True`) |
| LoRA r / alpha / dropout | 16 / 16 / 0 |
| Módulos-alvo | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Épocas / taxa de aprendizado / scheduler | 2 / 2e-4 / cosseno com aquecimento de 3% |
| Batch efetivo (batch × acumulação) / `max_seq_length` | 16 (2 × 8) / 2048 |
| Otimizador / precisão / seed | `adamw_8bit` / fp16 / 42 |
| GPU / tempo de treino | T4 / ___ |

<!-- fonte: colab/treino_log.csv -->
| Curva de treino | Valor |
|---|---|
| Passos de otimização | |
| Loss de treino: inicial / final | |
| Loss de validação: inicial / final / mínima (passo) | |

_Opcional: gráfico da loss por passo, gerado a partir de `colab/treino_log.csv`._

**Exportação:**

- o modelo foi exportado em GGUF `q4_k_m` (`models/qwen3-manual-q4_k_m.gguf`);
- foi registrado no Ollama com `ollama create qwen3-manual:4b -f Modelfile`;
- o Modelfile é o do `qwen3:4b` com só a linha `FROM` trocada (`colab/criar_modelo_local.sh`), então template de
  chat, parâmetros e tokens de parada são idênticos;
- quantização do leitor base no Ollama: ___ (a tag `qwen3:4b` costuma ser `Q4_K_M`; confira em
  `ollama show qwen3:4b`). Base e ajustado devem ter a mesma quantização para que a comparação seja justa.

## 5. Conjunto de avaliação

**Geração:**

- `qa-manual gerar-perguntas` pede 150 candidatas ao `llama3.2:3b`, um modelo distinto dos leitores.
- 20% das candidatas são `sem_resposta`, e as respondíveis se dividem em 60% factuais e 40% procedimentais.
- A amostra de trechos é estratificada por tema, com trechos de pelo menos 120 tokens.
- Cada candidata tem até 3 tentativas para produzir JSON válido.

**Revisão humana de 100% das candidatas:**

- o CSV vem de `exportar-revisao`, e a revisão segue [docs/protocolo_anotacao.md](protocolo_anotacao.md);
- respostas de referência têm até 20 palavras, com os termos do manual;
- a evidência é dada por ids `t####`;
- os itens `sem_resposta` são conferidos no manual inteiro.

**Divisão:** `importar-revisao --dividir` separa dev e teste (cerca de 30/70) por `tema_id`. Nenhuma seção aparece
nos dois conjuntos, e ambos recebem itens `sem_resposta`.

<!-- fonte: saída de `gerar-perguntas` no terminal (anotada durante a execução) -->
| Geração | Valor |
|---|---|
| Candidatas pedidas / geradas | |
| Tentativas / JSON válido (%) | |

<!-- fonte: data/gold_revisao.csv (colunas tipo, decisao e observacao) -->
| Revisão | factual | procedimental | sem_resposta | Total |
|---|---|---|---|---|
| aceitar | | | | |
| editar | | | | |
| descartar | | | | |
| `sem_resposta` convertidos em respondíveis | – | – | | |

Motivos de descarte mais frequentes: ___. Tempo de revisão: ___.

<!-- fonte: data/gold_dev.jsonl e data/gold_test.jsonl (campos tipo e tema_id) -->
| | Dev | Teste |
|---|---|---|
| factual | | |
| procedimental | | |
| sem_resposta | | |
| Total | | |
| Temas (seções) | | |

<!-- fonte: log de `importar-revisao` (aviso de redistribuição, se houver) -->
Os itens `sem_resposta` foram redistribuídos sem respeitar o tema? ___ (sim/não).

**Consistência entre revisores (opcional; protocolo, seção "Checagem de consistência"):**
<!-- fonte: planilha local de comparação -->
- itens em comum: ___;
- concordância em manter/descartar: ___% (κ = ___);
- concordância no tipo: ___%;
- concordância na evidência: ___%.

**Perplexidade:** 30 trechos de seções fora do teste foram reservados em `data/ppl_holdout.txt`
(`importar-revisao --reservar-ppl 30`).

## 6. Métricas

Todas as métricas são calculadas por `qa-manual avaliar`. EM, F1 e BERTScore comparam a resposta **sem a
citação** "(seção X, p. N)" com a referência. Uma abstenção num item respondível vale 0 nas três.

| Coluna de `metrics.csv` | Definição | Itens |
|---|---|---|
| `n` | registros avaliados | todos |
| `em` | exact match após normalização: minúsculas, sem pontuação, sem artigos e preposições frequentes, acentos mantidos, "1.420" igual a "1420" | respondíveis |
| `f1_tokens` | F1 sobre tokens normalizados, no estilo do SQuAD | respondíveis |
| `bertscore_f1` | BERTScore F1 (Zhang et al., 2020) com BERTimbau (`neuralmind/bert-base-portuguese-cased`), 9 camadas, sem reescala | respondíveis |
| `recall_at_5` | 1 se algum trecho da evidência está entre os 5 recuperados | respondíveis; vazio em S0 |
| `mrr_at_5` | inverso da posição da primeira evidência entre os 5 recuperados; 0 se ausente | respondíveis; vazio em S0 |
| `abst_correta` | fração de itens `sem_resposta` em que o sistema se absteve | `sem_resposta` |
| `abst_indevida` | fração de itens respondíveis em que o sistema se absteve | respondíveis |
| `alucinacao_sem_resposta` | 1 − `abst_correta`: o sistema respondeu algo quando devia abster-se | `sem_resposta` |
| `fonte_valida_pct` | fração das respostas não abstidas com citação que coincide com um trecho recuperado (mesma seção e página dentro do trecho) | não abstidas; 0 em S0 por construção |
| `latencia_mediana_s` / `latencia_p95_s` | tempo por pergunta, da recuperação à verificação, com o modelo já carregado | todos |
| `perplexidade` | perplexidade do leitor sobre os trechos reservados (seção 9) | por leitor |

Outras medidas:

- **`pareado.csv`:** ganho do ajustado sobre o base em EM e F1, pergunta a pergunta, com IC 95% por bootstrap
  (1000 reamostras, seed 42). Entram só os itens respondíveis respondidos pelos dois leitores na mesma
  configuração.
- **Gráficos:**
  - `respostas_por_config.png`: EM, F1 e BERTScore por configuração, com uma série para cada leitor;
  - `recall_por_config.png`: Recall@5 por configuração. A recuperação não depende do leitor.

## 7. Resultados

Conjunto de teste: ___ perguntas (___ respondíveis, ___ `sem_resposta`). Repetições: 1. Com temperatura 0 e seed
fixa, a saída é praticamente determinística.
<!-- fonte: runs/<run_id>/config.json (n_perguntas, repeticoes, gold_sha256) -->
Hash do gold de teste (12 primeiros caracteres): ___.

### 7.1 Qualidade da resposta e da recuperação

<!-- fonte: runs/<run_id>/metrics.csv (colunas config, leitor, n, em, f1_tokens, bertscore_f1, recall_at_5, mrr_at_5) -->
| config | leitor | n | em | f1_tokens | bertscore_f1 | recall_at_5 | mrr_at_5 |
|---|---|---|---|---|---|---|---|
| S0 | base | | | | | – | – |
| S0 | ajustado | | | | | – | – |
| S1 | base | | | | | | |
| S1 | ajustado | | | | | | |
| S2 | base | | | | | | |
| S2 | ajustado | | | | | | |
| S3 | base | | | | | | |
| S3 | ajustado | | | | | | |

### 7.2 Abstenção, citação e eficiência

<!-- fonte: runs/<run_id>/metrics.csv (colunas abst_correta, abst_indevida, alucinacao_sem_resposta, fonte_valida_pct, latencia_mediana_s, latencia_p95_s, perplexidade) -->
| config | leitor | abst_correta | abst_indevida | alucinacao_sem_resposta | fonte_valida_pct | latencia_mediana_s | latencia_p95_s | perplexidade |
|---|---|---|---|---|---|---|---|---|
| S0 | base | | | | – | | | |
| S0 | ajustado | | | | – | | | |
| S1 | base | | | | | | | |
| S1 | ajustado | | | | | | | |
| S2 | base | | | | | | | |
| S2 | ajustado | | | | | | | |
| S3 | base | | | | | | | |
| S3 | ajustado | | | | | | | |

<!-- fonte: runs/<run_id>/respostas_por_config.png -->
![EM, F1 e BERTScore por configuração e leitor](../runs/RUN_ID/respostas_por_config.png)

<!-- fonte: runs/<run_id>/recall_por_config.png -->
![Recall@5 por configuração](../runs/RUN_ID/recall_por_config.png)

_Opcional: decomposição da latência (embedding da pergunta, busca, leitura) e tokens de prompt e de resposta, a
partir dos campos `t_embedding_s`, `t_busca_s`, `t_leitura_s`, `n_tokens_prompt` e `n_tokens_resposta` de
`runs/<run_id>/respostas.jsonl`._

## 8. Ganho pareado do ajuste fino

<!-- fonte: runs/<run_id>/pareado.csv -->
| config | metrica | n_perguntas | media_base | media_ajustado | ganho_medio | ic95_inf | ic95_sup |
|---|---|---|---|---|---|---|---|
| S0 | em | | | | | | |
| S0 | f1_tokens | | | | | | |
| S1 | em | | | | | | |
| S1 | f1_tokens | | | | | | |
| S2 | em | | | | | | |
| S2 | f1_tokens | | | | | | |
| S3 | em | | | | | | |
| S3 | f1_tokens | | | | | | |

Um intervalo que não contém 0 indica diferença entre os leitores nessa configuração. _Discutir se o ganho, ou a
perda, se concentra em alguma configuração. Por exemplo, o formato de resposta aprendido pode ajudar mais quando
há trechos no prompt. Discutir também se acompanha a variação de `abst_indevida` e de `fonte_valida_pct`._

As comparações entre recuperadores (S1 × S2 × S3) são descritivas: o código não calcula intervalo de confiança
para elas.

## 9. Perplexidade

`qa-manual perplexidade` executa o `llama-perplexity` (llama.cpp) com contexto de 2048 tokens sobre os 30 trechos
reservados. Esses trechos são de seções fora do teste. Para o leitor base, usa-se o GGUF do próprio Ollama
(`--gguf-base ollama`).

<!-- fonte: runs/<run_ppl>/perplexidade.json (perplexidade, ctx, texto_sha256, gguf) -->
| Leitor | Modelo | GGUF | Perplexidade |
|---|---|---|---|
| base | `qwen3:4b` | blob do Ollama | |
| ajustado | `qwen3-manual:4b` | `models/qwen3-manual-q4_k_m.gguf` | |

Contexto: ___ tokens. Hash do texto reservado (12 primeiros caracteres): ___.

Como o ajustado nunca viu o manual, a perplexidade mede adaptação ao **domínio** portuário, e não memorização. Uma
perplexidade menor não implica respostas melhores: compare com a seção 8.

## 10. Análise de erros

**Amostra:**

- ___ itens do teste na configuração ___, para os dois leitores;
- entram os itens respondíveis com `f1_tokens` < 0,5 e os `sem_resposta` não abstidos;
- se houver mais de 50, faz-se um sorteio com seed 42.

**Classificação:** manual, comparando `recuperadas`, `resposta`, `absteve` e `fonte_valida` de
`runs/<run_id>/respostas.jsonl` com `evidencia` e `resposta_ref` do gold. Uma categoria por item, pela primeira que
se aplica na ordem da tabela.

<!-- fonte: classificação manual sobre runs/<run_id>/respostas.jsonl e data/gold_test.jsonl -->
| Categoria | Critério | n | % |
|---|---|---|---|
| Erro do gold | referência ou evidência errada ou incompleta | | |
| Falha de recuperação | nenhum trecho da evidência no top-5 | | |
| Abstenção indevida | item respondível, sistema se absteve com a evidência no top-5 | | |
| Falha de leitura | evidência no top-5, resposta errada ou incompleta | | |
| Alucinação em `sem_resposta` | o sistema respondeu algo quando devia abster-se | | |
| Correta, penalizada pela métrica | resposta certa com outra redação ou mais longa (EM 0, F1 baixo) | | |

Citação ausente ou inválida entre as respostas corretas: ___ de ___.

_Discutir dois exemplos parafraseados, um acerto e um erro, indicando a categoria e a configuração. Comentar casos
em que base e ajustado divergem na mesma pergunta._

## 11. Limitações

- **Gold:** é pequeno e produzido pelo grupo a partir de perguntas geradas por um LLM de 3B. Cada pergunta nasce de
  um único trecho, o que sub-representa perguntas que exigem combinar seções. Por isso os resultados vêm com
  intervalos de confiança.
- **Um único manual:** os resultados não se generalizam para outros documentos nem para outros domínios.
- **Heurística de títulos:** depende do layout (numeração no início da linha). Seções sem numeração herdam o tema
  anterior.
- **Tabelas:** tabelas complexas podem ser mal extraídas pelo PyMuPDF.
- **Parâmetros fixados a priori:** tamanho de trecho e k não foram otimizados.
- **Modelos:** leitores de 4B quantizados em `q4_k_m`. Em CPU, a latência limita o tamanho do leitor.
- **Ajuste fino:** os pares de treino foram gerados automaticamente a partir de um único documento público, sem
  revisão humana completa (ajustar se houve revisão).

## 12. Ameaças à validade

- **Interna:**
  - perguntas geradas a partir do texto de um trecho tendem a repetir as palavras dele, o que favorece o BM25. A
    revisão mitiga o problema só em parte.
  - a revisão foi feita por ___ revisor(es).
  - o Ollama em CPU não garante determinismo total, mesmo com temperatura 0 e seed fixa.
  - o mesmo `llama3.2:3b` gerou as perguntas do gold e os pares de treino. O ajustado pode ter aprendido o estilo
    dessas perguntas, e não só o domínio.
- **De construto:**
  - EM é rígido para um leitor generativo instruído a responder em frases. O F1 penaliza redações corretas e
    diferentes.
  - BERTScore sem reescala ocupa uma faixa estreita de valores: compare diferenças, não níveis.
  - `fonte_valida_pct` verifica se a citação coincide com um trecho recuperado, não se o trecho sustenta a
    resposta.
  - Recall@5 depende da completude da evidência anotada.
  - a perplexidade mede ajuste ao domínio, não qualidade de QA.
- **Externa:** um manual, um domínio, uma família de leitores (Qwen3 4B) e hardware de CPU específico.
- **De conclusão:**
  - o teste tem poucas perguntas (n = ___), e a divisão por tema reduz ainda mais o número por tipo;
  - os ICs por bootstrap cobrem só o ganho do ajustado;
  - as comparações entre recuperadores e as múltiplas comparações (4 configurações × 2 leitores) não têm correção.

## 13. Reprodução

```bash
qa-manual smoke --sem-rede
qa-manual indexar --manual data/manual.pdf
qa-manual stats
qa-manual gerar-perguntas --n 150
qa-manual exportar-revisao
# revisão de data/gold_revisao.csv (docs/protocolo_anotacao.md)
qa-manual importar-revisao --csv data/gold_revisao.csv --dividir --reservar-ppl 30
# Colab: QLoRA com o documento público -> models/qwen3-manual-q4_k_m.gguf
ollama create qwen3-manual:4b -f Modelfile
qa-manual perplexidade --binario <llama-perplexity> --gguf-base ollama \
  --gguf-ajustado models/qwen3-manual-q4_k_m.gguf
qa-manual avaliar --gold data/gold_test.jsonl        # S0-S3 x base/ajustado
```

`avaliar` usa o `perplexidade.json` mais recente em `runs/` e pode ser retomado com `--retomar <run_id>`. As
saídas ficam em `runs/<run_id>/`:

- `respostas.jsonl`;
- `metrics.csv`;
- `pareado.csv`;
- `respostas_por_config.png` e `recall_por_config.png`;
- `config.json`;
- `log.txt`.

## Referências

<!-- dados conferidos em outubro de 2026 (ACL Anthology, arXiv, Crossref, site do livro BPLN) -->

CORMACK, G. V.; CLARKE, C. L. A.; BÜTTCHER, S. Reciprocal rank fusion outperforms Condorcet and individual rank
learning methods. In: *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in
Information Retrieval*. ACM, 2009. p. 758–759.

CORTES, E. G.; VIEIRA, R.; BARONE, D. A. C. Perguntas e Respostas. In: CASELI, H. M.; NUNES, M. G. V. (org.).
*Processamento de Linguagem Natural: Conceitos, Técnicas e Aplicações em Português*. 2. ed. BPLN, 2024. cap. 16.
Disponível em: https://brasileiraspln.com/livro-pln/2a-edicao/parte-interacao/cap-qa/cap-qa.html.

DETTMERS, T.; PAGNONI, A.; HOLTZMAN, A.; ZETTLEMOYER, L. QLoRA: Efficient Finetuning of Quantized LLMs. In:
*Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, 2023. arXiv:2305.14314.

KARPUKHIN, V.; OĞUZ, B.; MIN, S.; LEWIS, P.; WU, L.; EDUNOV, S.; CHEN, D.; YIH, W. Dense Passage Retrieval for
Open-Domain Question Answering. In: *Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP)*. ACL, 2020. p. 6769–6781.

LEWIS, P.; PEREZ, E.; PIKTUS, A.; PETRONI, F.; KARPUKHIN, V.; GOYAL, N.; KÜTTLER, H.; LEWIS, M.; YIH, W.;
ROCKTÄSCHEL, T.; RIEDEL, S.; KIELA, D. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In:
*Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 2020. p. 9459–9474.

PERERA, M. M.; MAHMOOD, A.; WIJETHILAKE, K. E.; ISLAM, F.; TAHERMAZANDARANI, M.; SHENG, Q. Z. A Survey of the
State-of-the-Art in Conversational Question Answering Systems. arXiv:2509.05716, 2025.

ROBERTSON, S.; ZARAGOZA, H. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in
Information Retrieval*, v. 3, n. 4, p. 333–389, 2009.

ZAIB, M.; ZHANG, W. E.; SHENG, Q. Z.; MAHMOOD, A.; ZHANG, Y. Conversational question answering: a survey.
*Knowledge and Information Systems*, v. 64, n. 12, p. 3151–3195, 2022.

ZHANG, Q.; CHEN, S.; XU, D.; CAO, Q.; CHEN, X.; COHN, T.; FANG, M. A Survey for Efficient Open Domain Question
Answering. In: *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers)*. ACL, 2023. p. 14447–14465. DOI: 10.18653/v1/2023.acl-long.808.

ZHANG, T.; KISHORE, V.; WU, F.; WEINBERGER, K. Q.; ARTZI, Y. BERTScore: Evaluating Text Generation with BERT. In:
*International Conference on Learning Representations (ICLR)*, 2020.
