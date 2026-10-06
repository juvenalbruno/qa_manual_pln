# Protocolo de revisão do conjunto de avaliação

Guia para a revisão humana das perguntas candidatas geradas pelo `llama3.2:3b` (`qa-manual gerar-perguntas`) e
exportadas para planilha (`qa-manual exportar-revisao`). Leia tudo antes de começar: o gold é a régua da
avaliação, e um erro nele vira erro de métrica em todas as configurações.

## Fluxo

```bash
qa-manual gerar-perguntas --n 150     # data/gold_candidatas.jsonl (120 respondíveis + 30 sem_resposta)
qa-manual exportar-revisao            # data/gold_revisao.csv
# revisão do CSV, seguindo este protocolo
qa-manual importar-revisao --csv data/gold_revisao.csv --dividir --reservar-ppl 30
```

`importar-revisao` grava `data/gold_revisado.jsonl`. Com `--dividir`, separa `data/gold_dev.jsonl` (cerca de 30%)
e `data/gold_test.jsonl` (cerca de 70%) por `tema_id`: todas as perguntas de uma seção caem no mesmo conjunto.
Com `--reservar-ppl 30`, reserva 30 trechos de seções fora do teste para a medida de perplexidade. Se alguma
linha estiver incompleta ou inválida, o comando lista os problemas e não grava nada.

Cuidados:

- Revise **depois** da indexação final. Reindexar com outros parâmetros (`indexar --max-tokens ...`) muda os ids
  dos trechos e invalida a evidência já anotada.
- `exportar-revisao --forcar` sobrescreve o CSV e apaga a revisão feita. Guarde uma cópia em `data/` antes de
  reexportar.

## Confidencialidade

O CSV contém texto do manual: as perguntas, as respostas e a coluna `texto_evidencia`.

- Mantenha o arquivo em `data/`, que está fora do git e não pode ficar em pasta sincronizada (iCloud, Dropbox,
  OneDrive, Google Drive).
- Abra o arquivo no Excel ou no LibreOffice instalados na máquina. Não use Google Sheets nem Excel Online, e não
  envie o arquivo por e-mail ou mensageiro.
- Perguntas e respostas citadas no relatório ou nos slides devem ser **parafraseadas**.

## O arquivo de revisão

`data/gold_revisao.csv` usa `;` como separador e UTF-8 com BOM, para abrir direto no Excel em português e no
LibreOffice. Há uma linha por candidata.

| Coluna | Conteúdo | O revisor... |
|---|---|---|
| `id` | `q001`, `q002`, ... | não altera |
| `tipo` | `factual`, `procedimental` ou `sem_resposta` | corrige se o tipo estiver errado |
| `tema_id` | numeração da seção do trecho de origem (ex.: `4.2`) | altera só ao converter um `sem_resposta` (ver abaixo) |
| `evidencia` | ids dos trechos que contêm a resposta, separados por `\|` (ex.: `t0042\|t0043`) | completa ou corrige |
| `pergunta` | pergunta gerada | não altera; usa `pergunta_corrigida` |
| `resposta_ref` | resposta gerada (ou `Não encontrado no manual`) | não altera; usa `resposta_corrigida` |
| `texto_evidencia` | texto dos trechos da evidência; em `sem_resposta`, o trecho de origem com um aviso | só lê |
| `decisao` | `aceitar`, `editar` ou `descartar` | **preenche em todas as linhas** |
| `pergunta_corrigida` | nova redação da pergunta | preenche ao editar, se a pergunta mudar |
| `resposta_corrigida` | nova resposta de referência | preenche ao editar, se a resposta mudar |
| `observacao` | texto livre | motivo do descarte, conversões, dúvidas |

Como o importador trata as colunas:

- `decisao` vazia em qualquer linha impede a importação; o comando lista os ids pendentes.
- `editar` exige `pergunta_corrigida` ou `resposta_corrigida` preenchida. Coluna corrigida vazia mantém o texto
  original.
- `tipo`, `tema_id` e `evidencia` são lidos direto das colunas. Para mudar só o tipo ou a evidência, edite a coluna
  correspondente, use `aceitar` e registre a mudança em `observacao`.
- Itens `factual` e `procedimental` precisam de pelo menos um id de trecho em `evidencia`.
- Em itens `sem_resposta`, o importador grava a resposta `Não encontrado no manual` e esvazia a evidência, seja
  qual for o conteúdo dessas colunas.

> **Atenção ao Excel:** ao salvar, ele pode converter `tema_id` como `4.10` em número (`4,1`) ou em data, o que
> mistura seções na divisão dev/teste. Formate a coluna `tema_id` como texto antes de editar e confira alguns
> valores depois de salvar. Salve como "CSV UTF-8 (delimitado por vírgulas)"; o importador detecta `;`, `,` ou
> tabulação.

## Tipos de pergunta

| Tipo | Definição | Exemplo (fictício) |
|---|---|---|
| `factual` | Pede um fato pontual: valor, prazo, limite, nome, quantidade, responsável | "Qual a velocidade máxima de veículos no pátio?" |
| `procedimental` | Pede como fazer algo, a ordem dos passos ou a ação diante de uma situação | "O que o operador deve fazer ao detectar um vazamento?" |
| `sem_resposta` | Plausível para quem consulta o manual, mas o manual não contém a informação | "Qual o preço da armazenagem de carga refrigerada?" |

O gerador produz 60% factuais e 40% procedimentais entre as respondíveis, mas erra o tipo com frequência. Uma
pergunta "procedimental" cuja resposta é um único fato é `factual`.

## Decisões

**Aceitar** quando todos os critérios valem:

1. a pergunta é clara e autocontida (faz sentido sem ver o trecho);
2. pode ser respondida só com os trechos da evidência;
3. a resposta de referência está correta e segue as regras abaixo;
4. o tipo e a evidência estão corretos.

**Editar** quando a ideia é boa, mas a redação da pergunta ou a resposta precisam de ajuste. Prefira editar a
descartar quando a correção for rápida: cada descarte reduz o tamanho do gold.

**Descartar** quando a pergunta:

- é trivial (a resposta é o próprio título da seção ou está escrita na pergunta);
- é ambígua (admite mais de uma resposta correta no manual) ou depende de contexto ("nesse caso", "segundo o
  texto acima", "conforme a tabela");
- tem resposta errada que não dá para corrigir com o manual;
- trata de artefato de extração (cabeçalho, rodapé, sumário, número de página, texto truncado ou tabela
  embaralhada);
- exige conhecimento de fora do manual, como cálculo ou legislação não citada;
- repete outra candidata com outras palavras (fique com a melhor das duas);
- é `sem_resposta`, mas o manual responde, e a conversão não vale a pena (ver abaixo).

## Regras para a pergunta

1. Uma única pergunta por item. "Qual o prazo e quem autoriza?" deve ser editada para pedir uma coisa só, ou
   descartada.
2. Nomeie o objeto ("no pátio de contêineres"), em vez de "nesse local" ou "nessa etapa".
3. Não cite seção nem página na pergunta, porque isso facilitaria artificialmente a recuperação.
4. Ao editar, prefira a formulação de quem consulta o manual e evite copiar frases inteiras do trecho. Perguntas
   que repetem o texto favorecem o BM25.
5. Reescreva perguntas de sim/não como perguntas abertas, ou mantenha-as só se a resposta trouxer a informação
   ("Não; exige autorização do supervisor de turno").

## Regras para a resposta de referência

1. **Até 20 palavras**, em uma frase curta ou expressão nominal ("48 horas", "o supervisor de turno").
2. **Termos do manual:** valores, unidades, siglas e nomes de cargos aparecem como no manual. Não converta unidades
   nem reescreva valores. A métrica trata "1.420" e "1420" como iguais, mas não "48 h" e "48 horas".
3. **Sem citar seção nem página**, e sem "segundo o manual". O sistema cita a fonte, mas a citação é removida da
   resposta antes do EM, do F1 e do BERTScore.
4. **Só o que foi perguntado.** Palavras a mais na referência derrubam o F1 de respostas corretas.
5. **Procedimentos:** os passos essenciais na ordem do manual, separados por vírgula ou ponto e vírgula. Se não
   couberem em 20 palavras, a pergunta é ampla demais: restrinja-a ("Qual o primeiro passo...") ou descarte.
6. **Listas** (EPIs, documentos exigidos): todos os itens que o manual lista. Se passar de 20 palavras, restrinja
   a pergunta.
7. **`sem_resposta`:** a resposta é exatamente `Não encontrado no manual`. Não é preciso preencher
   `resposta_corrigida`, porque o importador grava essa frase.

## Evidência

- A evidência é a lista de ids de trecho (`t0001`, `t0002`, ...) de `data/trechos.jsonl` que **contêm a resposta**,
  separados por `|`. Não inclua trechos apenas relacionados ao assunto.
- O gerador anota só o trecho de origem. Confira se a resposta está mesmo nele e procure se ela também aparece em
  outro trecho: o Recall@5 conta acerto quando *qualquer* id da evidência está entre os 5 recuperados, então uma
  evidência incompleta subestima a recuperação.
- Para procurar outros trechos, use ferramentas locais:

  ```bash
  qa-manual recuperar --pergunta "<pergunta>" --modo bm25 --k 10
  qa-manual recuperar --pergunta "<pergunta>" --modo hibrido --k 10
  ```

  Cada linha mostra o id, o score, a seção, a página e o início do texto. Também vale buscar termos no PDF ou em
  `data/trechos.jsonl`.
- `tema_id` é o da seção do trecho que contém a resposta. Se a evidência estiver em seções diferentes, fique com a
  seção do trecho principal e registre isso em `observacao`.

## Itens `sem_resposta`

O gerador viu **um único trecho** e pediu um detalhe ausente dele. Esse detalhe pode estar em outra parte do
manual, então todo item `sem_resposta` precisa de checagem no manual inteiro:

1. leia a pergunta e o trecho de origem (`texto_evidencia`, marcado com "[trecho de origem; a resposta NÃO deve
   estar no manual]");
2. rode `qa-manual recuperar` com a pergunta nos modos `bm25` e `hibrido`, com `--k 10`, e leia os trechos;
3. busque as palavras-chave no PDF.

Um bom `sem_resposta`:

- é plausível para quem consulta o manual e pertence ao mesmo domínio;
- não se responde por inferência simples a partir do manual;
- não tem resposta parcial no manual. Por exemplo, se o manual dá uma faixa e a pergunta pede o valor exato, a
  pergunta é ambígua: reformule ou descarte.

Se a resposta for conhecimento geral (uma lei, uma norma) que **não** está no manual, o item continua
`sem_resposta`: o sistema deve responder só com o manual.

### Quando a pergunta `sem_resposta` tem resposta no manual

- **Converter em respondível**, se a pergunta for boa:
  - `decisao` = `editar`;
  - `tipo` = `factual` ou `procedimental`;
  - `evidencia` = ids dos trechos com a resposta;
  - `tema_id` = seção desses trechos (o `recuperar` mostra);
  - `resposta_corrigida` = resposta segundo as regras acima;
  - `observacao` = "convertida de sem_resposta".
- **Descartar**, se a pergunta for fraca ou a resposta do manual for parcial ou ambígua.

As duas saídas reduzem a proporção de `sem_resposta`, planejada em 20%. Se ela cair muito, acrescente linhas
novas escritas à mão: `id` novo (`m001`, `m002`, ...), `tipo` = `sem_resposta`, `tema_id` da seção mais próxima
do assunto, a pergunta na coluna `pergunta` e `decisao` = `aceitar`. Anote "escrita pelo revisor" em
`observacao`, porque o importador marca a origem desses itens como `gerado`. Elas passam pela mesma checagem no
manual inteiro.

## Checagem de consistência entre dois revisores (opcional)

O código importa um único CSV e não calcula concordância. Mesmo assim, uma checagem simples num subconjunto
mostra se o protocolo está claro e dá um número para o relatório.

1. **Subconjunto fixado antes da revisão:** as candidatas com número de id múltiplo de 5 (`q005`, `q010`, ...),
   cerca de 30 das 150.
2. **Revisão independente:** o revisor B trabalha numa cópia (`data/gold_revisao_B.csv`, na mesma máquina ou em
   outra autorizada a ter o manual) e preenche só as linhas do subconjunto, sem ver as decisões de A.
3. **Comparação numa planilha local**, linha a linha:
   - manter (`aceitar` ou `editar`) × `descartar`: tabela 2 × 2 e porcentagem de concordância;
   - tipo, entre as linhas que os dois mantiveram;
   - evidência, contando como concordância o mesmo conjunto de ids;
   - resposta: as referências de A e de B dizem a mesma coisa?
4. **Kappa de Cohen (opcional),** para manter × descartar: κ = (p_o − p_e) / (1 − p_e), em que p_o é a concordância
   observada e p_e = p_A(manter) · p_B(manter) + p_A(descartar) · p_B(descartar). Com cerca de 30 itens o kappa é
   instável; reporte-o junto com a tabela 2 × 2.
5. **Consenso:** as divergências são discutidas, e a decisão final é escrita no CSV de A, o único importado. Se uma
   divergência revelar uma regra ambígua, ajuste este protocolo e revise as linhas de A afetadas. Depois do
   consenso, apague a cópia de B ou mantenha-a em `data/`.

## Antes de importar

- [ ] Toda linha tem `decisao` (`aceitar`, `editar` ou `descartar`).
- [ ] Toda linha `editar` tem `pergunta_corrigida` ou `resposta_corrigida`.
- [ ] Todo item `factual` ou `procedimental` mantido tem ao menos um id `t####` em `evidencia`.
- [ ] Respostas de referência com até 20 palavras, sem seção nem página.
- [ ] Todos os `sem_resposta` foram conferidos no manual inteiro.
- [ ] `tema_id` não foi alterado pelo Excel e o arquivo foi salvo em UTF-8.

## O que registrar para o relatório

- A linha impressa por `gerar-perguntas` (candidatas geradas, pedidas e taxa de JSON válido). Ela não é gravada
  em arquivo.
- Contagem de `aceitar`, `editar` e `descartar` por tipo, tirada da coluna `decisao`, e os motivos de descarte mais
  frequentes, tirados de `observacao`.
- Número de `sem_resposta` convertidos em respondíveis e de itens escritos à mão.
- Tempo total de revisão.
- Se feita, a checagem entre revisores: número de itens, concordância em cada critério e kappa.
