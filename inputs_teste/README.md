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
| `manual_porto_salvador.pdf` | Manual de 18 páginas: capa, sumário, 22 capítulos, 4 tabelas, cabeçalho e rodapé em todas as páginas |
| `gold_dev.jsonl` | 47 perguntas de desenvolvimento (35 factuais, 4 procedimentais, 8 sem resposta) |
| `gold_test.jsonl` | 105 perguntas de teste (72 factuais, 16 procedimentais, 17 sem resposta) |
| `consultas_teste.jsonl` | 5 consultas com a passagem esperada, para `testar-indice` |
| `perguntas.txt` | 10 perguntas avulsas para `perguntar --arquivo`, algumas sem resposta no manual |
| `casos_de_erro/` | Entradas inválidas: PDF digitalizado sem texto, extensão errada, gold com campo ausente, tipo inválido, JSON quebrado e arquivo vazio |
| `fonte/` | Fontes editáveis: `manual_porto.html`, `capa.html` e `gold_fonte.yaml` (perguntas com âncoras) |
| `gerar_inputs.py` | Regenera tudo a partir de `fonte/` |

Dev e teste foram divididos **por seção** do manual (30/70), sem seção em comum. Os ids de evidência (`p0001`…)
correspondem ao índice criado pelo `indexar` com os parâmetros padrão de `configs/indexacao.yaml`
(400 tokens, sobreposição de 50). Se você mudar a segmentação, rode `gerar_inputs.py` de novo.

## Execução com um único comando

```bash
python -m src.cli executar \
  --manual     inputs_teste/manual_porto_salvador.pdf \
  --gold       inputs_teste/gold_test.jsonl \
  --gold-dev   inputs_teste/gold_dev.jsonl \
  --consultas  inputs_teste/consultas_teste.jsonl \
  --perguntas  inputs_teste/perguntas.txt
```

O comando valida todos os arquivos antes de começar e confere o Ollama e os modelos. Depois indexa o manual
(pula se o índice já estiver atualizado), testa o índice com as consultas e responde as perguntas avulsas com S3.
Em seguida avalia S0 a S3 no gold de teste e gera a análise (métricas, comparações, gráfico Recall@k e amostra
de erros). Opções úteis:

| Opção | Efeito |
|---|---|
| `--configs S0 S3` | escolhe as configurações (padrão: S0 S1 S2 S3) |
| `--repeticoes 3` | repete a avaliação e reporta média ± desvio |
| `--limite 30` | avalia só 30 perguntas (experimento mínimo) |
| `--sem-juiz` | pula o juiz local (mais rápido) |
| `--varrer` | varre tamanho de passagem × k no `--gold-dev` antes da avaliação |
| `--reindexar` | força a reindexação |

Só `--manual` e `--gold` são obrigatórios. Se forem omitidos, o comando pergunta os caminhos.

## Roteiro de teste passo a passo

```bash
# da raiz do repositório, com o ambiente ativado e o Ollama rodando
python smoke_test.py
python -m src.cli indexar --manual inputs_teste/manual_porto_salvador.pdf
python -m src.cli inspecionar --n 20
python -m src.cli testar-indice --consultas inputs_teste/consultas_teste.jsonl
python -m src.cli recuperar --pergunta "Quantos rebocadores precisa um navio de 340 m?" --modo hibrido
python -m src.cli perguntar --config S3 --arquivo inputs_teste/perguntas.txt

# experimento mínimo (rápido) e avaliação completa
python -m src.cli avaliar --gold inputs_teste/gold_test.jsonl --configs S0 S3 --limite 30
python -m src.cli avaliar --gold inputs_teste/gold_test.jsonl --configs S0 S1 S2 S3 --repeticoes 3
python -m src.cli analisar --run runs/<timestamp>_avaliar

# ajuste de parâmetros no dev
python -m src.cli varrer --gold inputs_teste/gold_dev.jsonl --manual inputs_teste/manual_porto_salvador.pdf

# mensagens de erro (todas devem sair com código 2)
python -m src.cli indexar --manual inputs_teste/casos_de_erro/manual_digitalizado.pdf
python -m src.cli indexar --manual inputs_teste/casos_de_erro/manual_extensao_errada.txt
python -m src.cli avaliar --gold inputs_teste/casos_de_erro/gold_tipo_invalido.jsonl --configs S0
```

Com 8 GB de RAM, use `--configs S3_qwen3_4b` para iterar mais rápido.

## Como alterar

1. Edite `fonte/manual_porto.html` (títulos `<h1>`/`<h2>` numerados viram seções) ou `fonte/gold_fonte.yaml`.
2. Cada pergunta com resposta precisa de uma `ancora`: um trecho literal do manual que identifica a passagem.
3. Rode `python inputs_teste/gerar_inputs.py`. O script falha e lista as âncoras que não encontrou.
4. Rode `pytest tests/test_inputs_teste.py` para conferir a consistência com o índice.
