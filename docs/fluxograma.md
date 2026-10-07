# Fluxograma do processo

Como o sistema funciona hoje, fase por fase. Os fluxos detalhados usam cores para a situação de cada etapa em 7 de
outubro de 2026:

| Cor | Situação |
|---|---|
| Verde | Verificada: testada com dados reais ou com lógica determinística conferida pelos testes |
| Amarelo | Pronta, mas testada só com simulação (depende de um modelo do Ollama que ainda não rodou) |
| Cinza tracejado | Pronta, ainda não executada (Colab e `llama-perplexity`) |
| Azul | Etapa humana |
| Vermelho | Recusa ou parada com mensagem que diz o que fazer |

## Visão geral

A máquina local faz tudo que envolve o manual; o Colab só recebe o documento público e devolve o modelo ajustado.

```mermaid
flowchart LR
  subgraph COLAB["Google Colab · GPU T4 · só documento público"]
    direction TB
    PUB[/"Regulamento portuário público"/] --> FT["Ajuste fino do leitor<br/>passos 5 a 7"]
  end
  subgraph LOCAL["Máquina local · CPU · o manual não sai daqui"]
    direction TB
    MAN[/"manual.pdf"/] --> IDX["Indexação<br/>passos 1 a 4"]
    IDX --> TR[("trechos e índices")]
    TR --> GOLD["Conjunto de avaliação<br/>passo 12 e revisão humana"]
    GOLD --> GS[("gold de dev e de teste")]
    PERG[/"Pergunta"/] --> CON["Consulta<br/>passos 8 a 11"]
    TR --> CON
    GS --> AVA["Avaliação<br/>passos 13 a 15"]
    AVA -->|"cada pergunta do gold"| CON
    CON --> RESP[("respostas.jsonl")]
    RESP --> AVA
    AVA --> MET[("metrics.csv, pareado.csv e gráficos")]
    OLL["Ollama local<br/>qwen3:4b · bge-m3 · llama3.2:3b"]
  end
  FT -->|"GGUF baixado vira qwen3-manual:4b"| OLL
  OLL -.->|"embeddings"| IDX
  OLL -.->|"gera perguntas"| GOLD
  OLL -.->|"leitor e embeddings"| CON
```

## Indexação (`qa-manual indexar`)

Do PDF aos índices. O PDF é lido antes de qualquer chamada ao Ollama.

```mermaid
flowchart TD
  A[/"manual.pdf"/] --> B{"Pasta sincronizada<br/>com a nuvem?"}
  B -->|sim| BX["Recusa: mova o projeto<br/>para uma pasta local"]
  B -->|não| C["1 · Extrai os blocos de texto<br/>PyMuPDF, ordem de leitura"]
  C --> D{"PDF tem camada de texto?"}
  D -->|não| DX["Para: rode OCR local<br/>com ocrmypdf"]
  D -->|sim| E["2 · Remove cabeçalho e rodapé<br/>repetidos em mais de 50% das páginas"]
  E --> F["Limpa o texto<br/>hifenização e quebras de linha"]
  F --> G{"A linha casa com a regex<br/>de título e a numeração<br/>está em sequência?"}
  G -->|sim| H["Abre um novo tema"]
  G -->|não| I["Parágrafo do tema atual<br/>descartado se tiver menos de 25 caracteres"]
  H --> P[("Parágrafos com tema e página")]
  I --> P
  P --> J["3 · Agrupa em trechos<br/>até 400 tokens, mesmo tema,<br/>sem cortar parágrafo"]
  J --> K[("data/trechos.jsonl<br/>trechos_stats.json")]
  K --> L["4a · Índice BM25<br/>bm25s"]
  K --> M["4b · Embeddings bge-m3 no Ollama<br/>índice FAISS"]
  L --> N[("index/bm25/")]
  M --> O[("index/dense.faiss")]
  N --> Q["Manifesto<br/>hash do PDF, dos trechos e parâmetros"]
  O --> Q
  Q --> R[("index/manifest.json")]

  classDef ok fill:#dcefe2,stroke:#2f7a4a,color:#12301d
  classDef sim fill:#f7ecc8,stroke:#9a7413,color:#3a2c05
  classDef erro fill:#f6dcdc,stroke:#a33a3a,color:#3d1111
  class C,E,F,H,I,J,L,Q ok
  class M sim
  class BX,DX erro
```

## Consulta (`qa-manual perguntar`, e cada pergunta dentro do `avaliar`)

Uma pergunta, da recuperação ao registro. Só o recuperador e o leitor mudam entre as configurações.

```mermaid
flowchart TD
  Q[/"Pergunta"/] --> V{"Índice existe e está atualizado?<br/>manifesto confere PDF, trechos e modelo"}
  V -->|não| VX["Recusa: rode indexar"]
  V -->|sim| S{"Configuração"}
  S -->|"S0 · sem recuperação"| P0["Prompt sem trechos<br/>leitor_s0.txt"]
  S -->|"S1 · BM25"| B["BM25: 5 trechos"]
  S -->|"S2 · denso"| D["bge-m3 + FAISS: 5 trechos"]
  S -->|"S3 · híbrido"| H["20 do BM25 + 20 do denso<br/>fusão RRF com c = 60<br/>5 trechos"]
  B --> L{"Limiar de score ligado<br/>e melhor score abaixo dele?"}
  D --> L
  H --> L
  L -->|sim| AB["Responde Não encontrado no manual<br/>sem chamar o leitor"]
  L -->|"não · padrão: limiar desligado"| P1["Prompt com os 5 trechos<br/>cada um com seção e página"]
  P0 --> R["10 · Leitor no Ollama<br/>base qwen3:4b ou ajustado qwen3-manual:4b<br/>temperatura 0, seed 42, até 200 tokens"]
  P1 --> R
  R --> T["Remove o raciocínio (think)<br/>se vier na resposta"]
  T --> VF["11 · Verificador<br/>abstenção, citações seção e página,<br/>fonte bate com um trecho recuperado"]
  AB --> VF
  VF --> OUT[("runs/run_id/respostas.jsonl<br/>log.txt só com ids, contagens e tempos")]

  classDef ok fill:#dcefe2,stroke:#2f7a4a,color:#12301d
  classDef sim fill:#f7ecc8,stroke:#9a7413,color:#3a2c05
  classDef erro fill:#f6dcdc,stroke:#a33a3a,color:#3d1111
  class P0,B,H,AB,P1,T,VF ok
  class D,R sim
  class VX erro
```

## Conjunto de avaliação e avaliação (`gerar-perguntas` até `avaliar`)

Do conjunto de perguntas às métricas. A revisão humana fica entre a geração e a divisão.

```mermaid
flowchart TD
  T[("data/trechos.jsonl")] --> AM["Amostra estratificada por tema<br/>só trechos com 120 tokens ou mais"]
  AM --> G1["llama3.2:3b gera pergunta e resposta<br/>80% factual ou procedimental (60/40)<br/>20% sem resposta no trecho"]
  G1 --> J{"Saída é JSON válido?"}
  J -->|"não · nova seed, até 3 tentativas"| G1
  J -->|sim| C[("gold_candidatas.jsonl")]
  C --> X["exportar-revisao<br/>gold_revisao.csv com o texto da evidência"]
  X --> HUM["Revisão humana no CSV<br/>aceitar, editar ou descartar"]
  HUM --> IM["importar-revisao --dividir<br/>temas inteiros: cerca de 30% dev, 70% teste"]
  IM --> DEV[("gold_dev.jsonl")]
  IM --> TEST[("gold_test.jsonl")]
  IM --> PPL[("ppl_holdout.txt<br/>30 trechos de temas fora do teste")]
  TEST --> AV["avaliar: leitor × configuração S0 a S3 × pergunta<br/>cada resposta é gravada na hora"]
  DEV -.->|"ajustes de k e tamanho de trecho"| AV
  AV --> RET{"Execução interrompida?"}
  RET -->|sim| RT["--retomar continua<br/>do ponto em que parou"]
  RT --> AV
  RET -->|não| MET["13 · Métricas<br/>EM, F1, BERTScore, Recall@5, MRR,<br/>abstenção, fonte válida, latência"]
  PPL --> PX["14 · llama-perplexity<br/>leitor base × ajustado"]
  PX --> MET
  MET --> OUT[("15 · metrics.csv, pareado.csv<br/>e dois gráficos PNG")]

  classDef ok fill:#dcefe2,stroke:#2f7a4a,color:#12301d
  classDef sim fill:#f7ecc8,stroke:#9a7413,color:#3a2c05
  classDef nao fill:#e6e6ea,stroke:#6b6b78,color:#24242b,stroke-dasharray:4 3
  classDef humano fill:#e3e8f7,stroke:#3d5aa8,color:#16224a
  class AM,X,IM,RT,MET ok
  class G1,AV sim
  class PX nao
  class HUM humano
```

## Ajuste fino no Colab (`colab/finetune_qlora.ipynb`)

O documento público vira pares de treino no formato do leitor; o modelo ajustado volta como GGUF.

```mermaid
flowchart TD
  PUB[/"Regulamento portuário público (PDF)"/] --> OK{"DOCUMENTO_E_PUBLICO<br/>confirmado no notebook?"}
  OK -->|não| STOP["Notebook para antes do upload"]
  OK -->|sim| P13["Passos 1 a 3 com o código do projeto<br/>embutido no notebook, sem GitHub"]
  P13 --> P5["5 · llama3.2:3b gera pares<br/>até 1500; 20% com alvo de abstenção<br/>resposta termina com a fonte do trecho"]
  P5 --> CT["Prompt do leitor com o trecho certo<br/>e 2 a 4 distratores do BM25"]
  CT --> DS[("treino.jsonl 95%<br/>validacao.jsonl 5%")]
  DS --> P6["6 · QLoRA com Unsloth sobre Qwen3-4B<br/>4 bits, r = 16, 2 épocas<br/>perda só nos tokens da resposta"]
  P6 --> SAN["Checagem em 10 respostas da validação<br/>citação e abstenção"]
  SAN --> P7["7 · Exporta GGUF q4_k_m"]
  P7 --> DL[/"Download: GGUF, treino_log.csv<br/>e metricas_validacao.json"/]
  DL --> LOC["Na máquina local: criar_modelo_local.sh<br/>Modelfile do qwen3:4b com outro FROM"]
  LOC --> OL[("Ollama: qwen3-manual:4b")]

  classDef ok fill:#dcefe2,stroke:#2f7a4a,color:#12301d
  classDef sim fill:#f7ecc8,stroke:#9a7413,color:#3a2c05
  classDef nao fill:#e6e6ea,stroke:#6b6b78,color:#24242b,stroke-dasharray:4 3
  classDef erro fill:#f6dcdc,stroke:#a33a3a,color:#3d1111
  class P13 ok
  class P5,CT sim
  class P6,SAN,P7,LOC nao
  class STOP erro
```

## O que falta executar

- Etapas em amarelo: rodar com o Ollama real, começando pelo experimento mínimo (S0 e S3, leitor base, 30
  perguntas) com o manual público de [exemplos/porto_salvador/](exemplos/porto_salvador/).
- Etapas em cinza: rodar o notebook no Colab e compilar o `llama-perplexity`.
