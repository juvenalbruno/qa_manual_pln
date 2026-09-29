# Protocolo de anotação do conjunto de avaliação

Guia para a revisão das candidatas geradas por `gerar-perguntas` (etapa 5). Os dois anotadores devem ler este
protocolo antes de começar, para que a concordância medida no subconjunto comum reflita o critério e não
interpretações diferentes.

## Tipos de pergunta

| Tipo | Definição | Exemplo (fictício) |
|---|---|---|
| `factual` | Pede um fato pontual: valor, prazo, limite, nome, especificação | "Qual a velocidade máxima no pátio?" |
| `procedimental` | Pede como fazer algo, a ordem de passos ou a ação diante de uma situação | "O que fazer em caso de vazamento?" |
| `sem_resposta` | Plausível para o domínio, mas o manual não contém a informação | "Qual o preço da armazenagem refrigerada?" |

## Decisões

- **Aprovar** (`a`): a pergunta é clara, respondível só com a passagem-evidência, e a resposta está correta e tem
  até 15 palavras.
- **Corrigir** (`e`): a ideia é boa, mas a redação, a resposta, o tipo ou a evidência precisam de ajuste.
- **Descartar** (`d`) quando a pergunta:
  - é trivial (a resposta é copiada literalmente do título da seção);
  - é ambígua (admite mais de uma resposta correta no manual);
  - depende do trecho ("segundo o texto acima…");
  - tem resposta errada que não dá para corrigir com a passagem;
  - é `sem_resposta`, mas a informação **está** no manual (confira as "passagens parecidas" mostradas).

## Regras para a resposta de referência

1. Curta (até 15 palavras), sem citar seção ou página.
2. Usar os termos do manual. Não parafrasear valores ou unidades.
3. Procedimentos com vários passos: resumir os passos essenciais na ordem do manual.
4. Itens `sem_resposta`: a resposta é exatamente `Não encontrado no manual` e a evidência fica vazia.

## Evidência

A evidência é a lista de ids de passagem (`p0042`) que contêm a resposta. Se a resposta também estiver em outra
passagem (por exemplo, por causa da sobreposição), inclua as duas.

## Concordância

O anotador B revisa com `--comum` o mesmo subconjunto determinístico de 30% que o anotador A. O comando
`concordancia` calcula o acordo e o kappa de Cohen para manter/descartar e para o tipo, o F1 médio entre as
respostas e a fração de evidências idênticas. Divergências são discutidas, e a versão de consenso entra no
arquivo do anotador listado primeiro em `dividir-gold --revisoes`.
