# HAI 21.03 · Análise de detecção

Dashboard do TCC de Samuel Eleotério, Engenharia de Computação da UNIFEI.
Reproduz o teste histórico com as previsões salvas da Regressão Logística, do
Isolation Forest com PCA e do ensemble OU.

## Abrir — opção mais simples

1. Extraia o ZIP completo para uma pasta.
2. Abra `painel/index.html` no seu navegador.
3. Use **Análise temporal** para explorar um episódio. Para uma explicação passo a passo, clique em **Roteiro do episódio 08**.

A página tem os dados incorporados e funciona offline. Não precisa instalar
Python nem enviar os arquivos para um servidor para usar essa opção.
É necessário um navegador atualizado com suporte à leitura de gzip
(`DecompressionStream`).

## As três abas

**Análise temporal**

- A faixa de status mostra o rótulo do HAI e a resposta de cada modelo no segundo selecionado.
- Verde: ataque sinalizado. Vermelho: ataque sem alerta. Laranja: falso alarme. Cinza/azul: operação normal sem alerta ou informação da base.
- Os resultados também têm palavras e símbolos, para não depender só da cor.
- A faixa clara do gráfico marca um trecho rotulado como ataque. A linha azul é o sinal escolhido. Os pontos são alertas em ataque; os losangos são falsos alarmes.
- Arraste a linha do tempo, clique no gráfico ou use os botões de reprodução e avanço de um segundo.
- A comparação à direita mostra o primeiro alerta e a fração de registros sinalizados em cada episódio.
- O seletor de sinal muda apenas a visualização. As previsões continuam sendo as que foram calculadas com as entradas de cada modelo.
- A tabela **Registros de detecção** permite filtrar por modelo e resultado: todos os alertas, VP, FP ou FN. Mostra os 12 registros correspondentes mais recentes, até o instante selecionado, dentro da janela de análise. Clique em **Ver instante** para localizar um deles no gráfico.
- No modo **Teste completo**, a busca da tabela abrange o período desde o início do test5 até o cursor. O gráfico mostra uma janela de 240 segundos em torno do cursor. Nas janelas por episódio, o gráfico e a tabela incluem 30 segundos antes e depois do ataque.
- Os registros da tabela não são agrupados em incidentes. Os filtros apenas mudam a consulta; não alteram decisões nem limiares.

**Roteiro do episódio 08** usa momentos reais: operação normal,
início do ataque, primeiro alerta do ensemble, primeiro alerta da Logística e
um falso alarme. Na execução incluída, os primeiros alertas são aos 13 s e aos
40 s, respectivamente. Se importar outras previsões, o roteiro é recalculado;
um passo sem exemplo correspondente é omitido. Navegar manualmente encerra o
roteiro.

**Desempenho**

- Mostra os registros de ataque recuperados pelo ensemble e os falsos alarmes adicionados, em relação à Logística.
- As barras de ataques sinalizados e falsos alarmes têm escalas próprias, identificadas acima delas.
- Alterna a tabela entre métricas por registro e por episódio.
- O mapa dos 12 episódios mostra a cobertura de cada modelo. Clique em uma linha para abrir aquele trecho.
- O gráfico por hora permite localizar e abrir o primeiro falso alarme da hora escolhida.
- **Exportar métricas · CSV** exporta a execução em uso, com VP, FP, FN, VN, precisão, recall, F1, episódios e atraso mediano. Usa ponto e vírgula como separador e vírgula decimal.

**Referência**

Explica VP, VN, FP, FN, precisão, recall, F1, a regra OU e a diferença entre
registro e episódio. Os exemplos numéricos são recalculados a partir da
execução em uso. Os 51 quadradinhos do episódio 08 representam seus registros
de ataque: um quadradinho verde significa um registro sinalizado pela Logística.

## Fonte dos dados: atenção aos números do artigo

O pacote inclui a execução de referência `ensemble_20261112_215002_433402`.
Nela, a Logística tem **647 FP** e o ensemble, **1.121 FP**. A execução de Samuel
usada no workshop e na tabela principal do artigo é
`ensemble_20261112_215609_211220`, com **643 FP** e **1.118 FP**.

Para exibir a execução do artigo, abra **Fonte dos dados** e selecione
`teste_historico_ensemble_previsoes.csv.gz` (ou CSV descompactado) de:

```
runs/ensemble_20261112_215609_211220/previsoes_e_metricas/
```

O painel confere as 92.401 linhas, índices, horários, rótulos, episódios,
decisões binárias e a regra OU antes de aplicar o arquivo. Arquivos inválidos
não substituem a execução atual. O arquivo importado fica somente na sessão
aberta do navegador; ao reabrir a página, volta a referência incluída.
A origem aparece no alto da tela e nas métricas exportadas.

## Visual desta versão

Fundo claro, azul restrito aos controles e à série do gráfico, navegação lateral
e tipografia compacta. A tela principal prioriza a análise temporal, a comparação
dos modelos e a consulta aos registros. As explicações ficam em **Referência**;
o roteiro continua disponível como opção. O layout se reorganiza em telas menores.

## Usar no Streamlit

Dentro da pasta que contém `app.py`:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Se já utiliza um ambiente virtual no projeto, use o Python desse ambiente.
A interface é a mesma do HTML portátil. Os arquivos de `interface/` são o
código-fonte; após editá-los, execute `python montar_painel.py` para reconstruir
`painel/index.html`.

## Como interpretar

- Um registro é uma linha de medições em um segundo. Um episódio é um trecho contínuo rotulado como ataque.
- Sinalizar um episódio exige ao menos um alerta dentro do trecho; não implica identificar todos os seus registros.
- A mediana do atraso considera somente episódios com alerta. Ausência de alerta não tem atraso zero.
- FP e FN contam registros, não incidentes independentes.
- Uma variação no sinal não comprova sozinha um ataque. O rótulo do HAI é usado para conferir as previsões, não como entrada do modelo.
- O painel reproduz dados históricos e previsões prontas; não está conectado a uma instalação industrial e não treina modelos.
- O painel interativo contém os três conjuntos de previsões incluídos no pacote. As comparações completas com Random Forest e IF sem PCA estão na planilha de métricas do TCC.

## Verificação desta versão

Os arquivos de medições e previsões foram preservados byte a byte. Foram
conferidos os totais e os 12 episódios. Onze grupos de verificações programáticas
com DOM simulado cobrem métricas, filtros da tabela, limite temporal do cursor,
troca de episódio, reprodução, navegação, roteiro, validação de CSV, importação,
restauração da fonte e exportação. A tabela não antecipa registros futuros.
As verificações não substituem uma inspeção visual no navegador, que não pôde
ser concluída no ambiente de edição.

## Dados e créditos

Medições e rótulos do `test5` do HAI 21.03. As nove séries para visualização
preservam os valores do CSV e os horários da coleta.

- [Base HAI e documentação](https://github.com/icsdataset/hai)
- [Licença CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- [Shin et al., CSET 2021](https://doi.org/10.1145/3474718.3474719)

As previsões históricas são do Notebook 06. O ensemble alerta quando a
Logística ou o IF com PCA sinaliza o registro.
