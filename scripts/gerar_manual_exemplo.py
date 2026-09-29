"""Gera um manual fictício em PDF para testes e demonstração (não contém dados da empresa).

Uso: python scripts/gerar_manual_exemplo.py <saida.pdf>

O PDF tem títulos numerados, cabeçalho e rodapé repetidos, uma tabela com bordas
e uma palavra hifenizada em fim de linha, para exercitar a limpeza da etapa 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

CSS = """
body { font-family: sans-serif; font-size: 10pt; }
h1 { font-size: 16pt; font-weight: bold; margin-top: 14pt; }
h2 { font-size: 13pt; font-weight: bold; margin-top: 10pt; }
p { margin-bottom: 6pt; text-align: justify; }
table { border-collapse: collapse; }
td, th { border: 1px solid black; padding: 3pt; font-size: 9pt; }
"""

HTML = """
<h1>1 Introdução</h1>
<p>Este manual descreve os procedimentos operacionais do Terminal Portuário Exemplo, abrangendo o recebimento de navios,
a movimentação de contêineres no pátio, a liberação de cargas e as regras de segurança aplicáveis a colaboradores,
transportadores e visitantes. As instruções aqui contidas são obrigatórias para todas as equipes do terminal.</p>
<p>O manual é revisado anualmente pela Coordenação de Operações. Sugestões de melhoria devem ser enviadas por meio do
formulário interno FO-12, que é analisado em até 30 dias. Em caso de divergência entre este manual e a legislação vigente,
prevalece a legislação.</p>

<h1>2 Atracação de navios</h1>
<h2>2.1 Programação de atracação</h2>
<p>A programação de atracação é publicada diariamente às 10 horas no sistema de gestão portuária. O agente marítimo deve
confirmar a chegada do navio com antecedência mínima de 72 horas, informando calado, comprimento total e número de
contêineres a descarregar. Solicitações recebidas após esse prazo entram no fim da fila de programação.</p>
<p>Navios com calado superior a 12,5 metros só podem atracar no Berço 3, que possui a maior profundidade do terminal.
A manobra de atracação exige a presença de dois rebocadores para navios com mais de 200 metros de comprimento.</p>
<h2>2.2 Documentação exigida</h2>
<p>Antes da atracação, o agente marítimo deve apresentar o manifesto de carga, a lista de tripulantes, o certificado de
arqueação e a declaração marítima de saúde. A falta de qualquer documento impede a liberação do berço pela autoridade
portuária. Os documentos devem ser enviados em formato eletrônico pelo portal do terminal.</p>

<h1>3 Movimentação no pátio</h1>
<h2>3.1 Regras de circulação</h2>
<p>A velocidade máxima de veículos no pátio é de 20 km/h, reduzida para 10 km/h nas áreas de empilhamento. Pedestres
devem circular exclusivamente pelas faixas demarcadas em amarelo. É proibido o uso de telefone celular por operadores de
equipamentos em movimento.</p>
<p>Os caminhões externos devem permanecer com o motorista dentro da cabine durante toda a operação de carga e descarga,
exceto quando orientados pelo conferente. O tempo máximo de permanência de caminhões externos no pátio é de 4 horas.</p>
<h2>3.2 Empilhamento de contêineres</h2>
<p>A altura máxima de empilhamento é de cinco contêineres cheios ou sete contêineres vazios. Contêineres com carga perigosa
devem ser armazenados na área segregada D, com afastamento mínimo de 3 metros entre pilhas. Contêineres refrigerados
devem ser conectados à tomada em até 30 minutos após a descarga.</p>
<table>
<tr><th>Tipo de contêiner</th><th>Altura máxima</th><th>Área</th></tr>
<tr><td>Cheio (dry)</td><td>5 unidades</td><td>A e B</td></tr>
<tr><td>Vazio</td><td>7 unidades</td><td>C</td></tr>
<tr><td>Refrigerado</td><td>4 unidades</td><td>R</td></tr>
<tr><td>Carga perigosa</td><td>3 unidades</td><td>D</td></tr>
</table>

<h1>4 Liberação de carga</h1>
<h2>4.1 Prazos</h2>
<p>A liberação da carga ocorre em até 48 horas após a atracação, desde que todos os tributos estejam recolhidos e não haja
exigência de inspeção pela Receita Federal. Cargas selecionadas para canal vermelho podem levar até cinco dias úteis para
serem liberadas, conforme a disponibilidade da fiscalização.</p>
<p>O importador pode acompanhar o status da liberação pelo portal do terminal, utilizando o número do conhecimento de
embarque. A armazenagem é gratuita nos primeiros 7 dias; a partir do oitavo dia, aplica-se a tarifa de armazenagem
vigente.</p>
<h2>4.2 Retirada da carga</h2>
<p>Para retirar a carga, o transportador deve agendar a janela de retirada com antecedência mínima de 24 horas. No dia
da retirada, o motorista apresenta a ordem de coleta, a CNH e o documento do veículo na portaria principal. Caminhões
sem agendamento não têm acesso ao terminal.</p>
<p>A conferência da carga é feita pelo conferente do terminal na presença do motorista. Divergências de lacre ou avarias
devem ser registradas no termo de avaria antes da saída do veículo, sob pena de perda do direito de reclamação.</p>

<h1>5 Segurança do trabalho</h1>
<h2>5.1 Equipamentos de proteção individual</h2>
<p>O uso de capacete, colete refletivo, calçado de segurança e protetor auricular é obrigatório em todas as áreas
operacionais. Nas operações com carga perigosa, exige-se também luva de proteção química e óculos de segurança. O
colaborador que se recusar a usar os EPIs deve ser afastado da atividade pelo supervisor.</p>
<h2>5.2 Emergências</h2>
<p>Em caso de emergência, o colaborador deve acionar o alarme mais próximo e comunicar a Central de Segurança pelo ramal
190. O ponto de encontro principal fica no estacionamento da portaria principal. Simulados de evacuação são realizados a
cada seis meses com participação obrigatória de todas as equipes.</p>
<p>Vazamentos de produtos perigosos devem ser isolados em um raio mínimo de 50 metros até a chegada da brigada de
emergência. Somente a brigada pode realizar a contenção do vazamento, utilizando o kit de emergência ambiental.</p>

<h1>6 Manutenção de equipamentos</h1>
<p>Os portêineres passam por manu-<br/>tenção preventiva a cada 500 horas de operação, e as empilhadeiras a cada 250 horas.
A manutenção corretiva deve ser solicitada pelo operador por meio de ordem de serviço no sistema de manutenção, com
descrição do defeito observado.</p>
<p>Antes do início de cada turno, o operador deve realizar o checklist diário do equipamento, verificando freios, luzes,
buzina, nível de óleo e sistema hidráulico. Equipamentos reprovados no checklist ficam bloqueados até a liberação pela
equipe de manutenção.</p>
"""

CABECALHO = "Manual de Operações – Terminal Portuário Exemplo"


def gerar(saida: Path) -> Path:
    saida.parent.mkdir(parents=True, exist_ok=True)
    temporario = saida.with_suffix(".tmp.pdf")
    story = pymupdf.Story(html=HTML, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(temporario))
    pagina = pymupdf.paper_rect("a4")
    area = pagina + (60, 80, -60, -80)
    mais = True
    while mais:
        dispositivo = writer.begin_page(pagina)
        mais, _ = story.place(area)
        story.draw(dispositivo)
        writer.end_page()
    writer.close()

    doc = pymupdf.open(temporario)
    total = doc.page_count
    for i, pg in enumerate(doc, 1):
        pg.insert_text((60, 40), CABECALHO, fontsize=8)
        pg.insert_text((260, pg.rect.height - 30), f"Página {i} de {total}", fontsize=8)
    doc.save(saida)
    doc.close()
    temporario.unlink()
    return saida


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    print(gerar(Path(sys.argv[1])))
