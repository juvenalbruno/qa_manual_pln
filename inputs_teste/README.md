# Inputs de teste: Manual de Operações do Porto de Salvador

Arquivos de entrada para testar o sistema de QA de ponta a ponta sem usar o manual confidencial da empresa.
O manual de teste reúne informações **públicas e verificadas** sobre o Porto de Salvador (CODEBA), o terminal
de contêineres TECON Salvador, a navegação na Baía de Todos-os-Santos, o despacho aduaneiro, a segurança do
trabalho portuário (NR-29), as cargas perigosas (IMDG) e o meio ambiente. As fontes estão em [fontes.md](fontes.md).

> Não é publicação oficial da CODEBA, da Wilson Sons, da Marinha ou de qualquer órgão. Valores podem estar
> desatualizados, e as regras marcadas como "regra interna" foram criadas para o teste.

## Arquivos

| Arquivo | Uso |
|---|---|
| `manual_porto_salvador.pdf` | Manual de 18 páginas: capa, sumário, 22 capítulos, 4 tabelas, cabeçalho e rodapé em todas as páginas. Com `configs/base.yaml`, vira 59 trechos, um por seção |
| `gold_dev.jsonl` | 46 perguntas de desenvolvimento em 16 temas (34 factuais, 4 procedimentais, 8 sem resposta) |
| `gold_test.jsonl` | 105 perguntas de teste em 42 temas (72 factuais, 16 procedimentais, 17 sem resposta) |
| `perguntas.txt` | 10 perguntas avulsas para `qa-manual perguntar --arquivo`, algumas sem resposta no manual |
| `casos_de_erro/` | Entradas inválidas: PDF digitalizado sem texto, extensão errada, gold com campo ausente, tipo inválido, JSON quebrado e arquivo vazio |
| `fonte/` | Fontes editáveis: `manual_porto.html`, `capa.html` e `gold_fonte.yaml` (perguntas com âncoras) |
| `gerar_inputs.py` | Regenera tudo a partir de `fonte/` |

Cada linha do gold segue o contrato de `qa_manual/gold.py`:

```json
{"id": "q003", "pergunta": "Quando o Porto de Salvador foi inaugurado como porto organizado?", "resposta_ref": "13 de maio de 1913", "evidencia": ["t0003"], "tipo": "factual", "tema_id": "2.1", "origem": "manual", "revisado": true, "split": "dev"}
```

- As perguntas foram escritas à mão (`origem: "manual"`, `revisado: true`).
- `evidencia` lista os ids de trecho (`t0001`, ...) gravados em `data/trechos.jsonl` pelo `qa-manual indexar`
  com os parâmetros padrão de `configs/base.yaml` (`trechos.max_tokens: 400`, sem sobreposição). Se você mudar
  `ingestao` ou `trechos` (ou usar `--max-tokens`), os ids mudam: rode `gerar_inputs.py` de novo.
- `tema_id` é a seção do trecho-evidência. Nos itens `sem_resposta`, que têm `evidencia: []` e
  `resposta_ref: "Não encontrado no manual"`, o `tema_id` vem do próprio `gold_fonte.yaml`: é a seção mais
  próxima do assunto da pergunta.
- Dev e teste foram divididos **por tema** com `qa_manual.gold.dividir` (`frac_dev` 0,30, `seed` 42), sem tema
  em comum entre os dois.

## Roteiro de teste

Da raiz do repositório, com o ambiente ativado (`pip install -e .`), o Ollama rodando e os modelos `qwen3:4b` e
`bge-m3` baixados:

```bash
qa-manual smoke --sem-bertscore
qa-manual indexar --manual inputs_teste/manual_porto_salvador.pdf --permitir-pasta-sincronizada
qa-manual stats
qa-manual recuperar --pergunta "Qual o limite de vento para manobras de atracação em Salvador?" --modo bm25
qa-manual recuperar --pergunta "Como devo agendar a entrada de um caminhão no TECON?" --modo hibrido
qa-manual perguntar --config-exp S3 --leitor base --pergunta "Qual a profundidade do Berço 202?"
qa-manual perguntar --config-exp S3 --leitor base --arquivo inputs_teste/perguntas.txt

# experimento mínimo (rápido) e avaliação completa no gold de teste
qa-manual avaliar --gold inputs_teste/gold_test.jsonl --configs S0 S3 --leitores base --limite 30
qa-manual avaliar --gold inputs_teste/gold_test.jsonl --configs S0 S1 S2 S3 --leitores base

# ajuste de parâmetros (k, limiar de abstenção, prompt) só no dev
qa-manual avaliar --gold inputs_teste/gold_dev.jsonl --configs S0 S1 S2 S3 --leitores base
```

**Sobre `--permitir-pasta-sincronizada`.** O `indexar` recusa gravar quando o manual, `data/`, `index/` ou
`runs/` estão em pasta sincronizada com a nuvem (iCloud na Mesa ou em Documentos, Dropbox, OneDrive, Google
Drive). A flag é aceitável aqui porque o manual de teste é público e enviá-lo à nuvem não expõe nada, mas é
proibida com o manual real, porque o PDF e seus derivados (trechos, índices e respostas) sairiam da máquina e
violariam a regra de confidencialidade. Com o manual real, mova o projeto para uma pasta local.

Notas:

- `indexar` grava `data/trechos.jsonl`, `data/trechos_stats.json` e `index/` no diretório atual, substituindo um
  índice anterior. `perguntar` e `avaliar` usam esse índice.
- `avaliar` grava em `runs/<run_id>/` as respostas, `metrics.csv`, `pareado.csv`, os gráficos e o `log.txt`. Uma
  execução interrompida continua com `--retomar <run_id>`, e `--repeticoes 3` repete cada pergunta.
- Com o leitor ajustado criado no Ollama (`qwen3-manual:4b`), use `--leitores base ajustado` (ou omita
  `--leitores`) para comparar os dois leitores.

## Casos de erro

Todos devem sair com código 2 e uma mensagem que diz o que fazer (`echo $?` mostra o código):

```bash
qa-manual indexar --manual inputs_teste/casos_de_erro/manual_digitalizado.pdf --permitir-pasta-sincronizada
qa-manual indexar --manual inputs_teste/casos_de_erro/manual_extensao_errada.txt
qa-manual avaliar --gold inputs_teste/casos_de_erro/gold_campo_ausente.jsonl --configs S0 --leitores base
qa-manual avaliar --gold inputs_teste/casos_de_erro/gold_tipo_invalido.jsonl --configs S0 --leitores base
qa-manual avaliar --gold inputs_teste/casos_de_erro/gold_json_quebrado.jsonl --configs S0 --leitores base
qa-manual avaliar --gold inputs_teste/casos_de_erro/gold_vazio.jsonl --configs S0 --leitores base
```

| Caso | Mensagem esperada |
|---|---|
| `manual_digitalizado.pdf` | `PDF sem texto; rode OCR local (ocrmypdf) antes` |
| `manual_extensao_errada.txt` | `Extensão inválida ... esperado .pdf.` |
| `gold_campo_ausente.jsonl` | `gold inválido: q001: campo(s) ausente(s): resposta_ref, evidencia` |
| `gold_tipo_invalido.jsonl` | `gold inválido: q001: tipo 'opinativa' inválido` |
| `gold_json_quebrado.jsonl` | `linha 1: JSON inválido` |
| `gold_vazio.jsonl` | `gold vazio.` |

O caso do PDF digitalizado precisa da flag `--permitir-pasta-sincronizada` para que a recusa por pasta
sincronizada não apareça antes; o PDF é lido antes de qualquer chamada ao Ollama, então o erro aparece mesmo com o
Ollama desligado. Os seis casos também são conferidos por `tests/test_inputs_teste.py`, com o cliente Ollama falso.

## Limitações conhecidas da extração

A ingestão junta as células de cada linha de tabela com espaços (`202 Armazéns 3 e 4 11,5 m 300 m 50.000 TPB`) e
descarta blocos com menos de 25 caracteres (`ingestao.min_chars_paragrafo`). Por isso as linhas da tabela de
rebocadores (seção 5.2) e algumas linhas da tabela IMDG (seção 13.1, como `1 Explosivos 1.1 a 1.6`) não entram
nos trechos. O gold não tem perguntas que dependam desse texto, e a pergunta avulsa sobre rebocadores para um
navio de 340 m deve terminar em abstenção.

## Como alterar

1. Edite `fonte/manual_porto.html` (títulos `<h1>`/`<h2>` numerados viram temas) ou `fonte/gold_fonte.yaml`.
2. Cada pergunta respondível precisa de uma `ancora`: um trecho literal do **texto extraído** (sem diferenciar
   maiúsculas e espaços) que identifica o trecho-evidência. Linhas de tabela usam espaços entre as células, não
   `|`. Cada pergunta `sem_resposta` precisa de um `tema_id` entre aspas (`"5.1"`) que exista no manual.
3. Rode `.venv/bin/python inputs_teste/gerar_inputs.py`. O script falha e lista as âncoras não encontradas (ou
   encontradas em temas diferentes) e os `tema_id` inválidos.
4. Rode `.venv/bin/pytest tests/test_inputs_teste.py` para conferir o gold contra os trechos atuais.
