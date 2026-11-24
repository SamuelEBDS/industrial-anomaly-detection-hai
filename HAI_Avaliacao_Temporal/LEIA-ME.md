# Os resultados se repetem em outros períodos do HAI?

Estudo complementar à comparação do TCC. A execução já está pronta: abra o PDF ou o notebook para ler os resultados, sem treinar nada. O pacote inclui os métodos principais: Random Forest (RF) com pesos de classe, Isolation Forest (IF) sem PCA, IF com PCA, Regressão Logística e a regra OU da Logística com o IF com PCA. As variantes de subamostragem ou de baixa vazão não fazem parte desta pergunta.

## O que mudou?

| Rodada | Treino | Validação do limiar | Teste | Entradas |
|---|---|---|---|---|
| J1 | 21.03/test1 | 21.03/test2 | 21.03/test3 (8 episódios) | Variáveis não constantes no treino: 58 |
| J2 | 21.03/test1–2 | 21.03/test3 | 21.03/test4 (5) | 60 |
| J3 | 21.03/test1–3 | 21.03/test4 | 21.03/test5 (12) | 60 |
| EXT | 21.03/test1–3 | 21.03/test4 | 21.03/test5 como referência e os quatro arquivos de teste do 22.04 (58 episódios externos) | 52 sinais com o mesmo nome nas duas versões |

J3 repete o desenho do artigo **nesta execução**. J1 e J2 acrescentam duas divisões; apenas os testes das três divisões internas são disjuntos. Um arquivo de teste de uma rodada pode ser usado no treino ou na validação **de uma rodada cronologicamente posterior**. As rodadas compartilham dados de treino; não são três amostras independentes. O treino também ganha volume e exemplos de ataque: a variação de F1 não pode ser atribuída unicamente à data do teste.

Para EXT, apenas os cabeçalhos externos são consultados para saber quais sinais existem. Todos os modelos são treinados de novo com as mesmas 52 entradas; não se preenchem sinais ausentes com zeros. Os parâmetros e limiares são definidos sem usar as medições ou os rótulos de 22.04. As previsões no test5 do 21.03 com 52 sinais dão uma referência comparável à transferência. Os quatro arquivos do 22.04 são quatro períodos, mas pertencem à mesma versão da bancada. O RF já havia sido estudado nessa versão: este complemento é retrospectivo, não uma coleta inédita ou um experimento cego. Igualdade de nomes não garante igualdade de escala, instrumentação ou condições físicas.

## Como os modelos geram os alertas

- Só entram sinais industriais. Excluem-se horário, metadados e rótulos; só o treino determina quais sinais são constantes. Os sinais são alinhados pelo **nome** e na ordem do treino.
- RF: 100 árvores, `balanced_subsample`, semente 42. Regressão Logística: treino com normais e ataques, `StandardScaler` no treino inteiro, penalização L2, C=1, solver lbfgs, tolerância 1e-6 e máximo de 1.000 iterações. A falta de convergência interrompe o programa.
- IF: padronização ajustada apenas nas linhas normais do treino, 100 árvores, amostra de até 256 por árvore, contaminação automática e semente 42. IF com PCA: os mesmos dados normais para ajustar a padronização e o PCA (variância de pelo menos 95%, SVD completo). Escore de anomalia = negativo de `score_samples`.
- Em cada rodada, cada método individual escolhe o limiar de maior F1 na validação. Em empate mantém o maior limiar. Alerta se escore ≥ limiar. RF e Logística usam a probabilidade da classe de ataque. O ensemble gera alerta quando pelo menos um de seus dois componentes gera alerta. Não há um quinto limiar. O código grava sinais e limiares antes da avaliação da rodada.
- Nada aqui reestima pesos de classe a partir do teste, troca regras depois dos resultados ou mede o efeito de novas sementes. Os três sorteios 13/42/2026 pertencem ao outro experimento, de subamostragem.

## O que significam os números

As métricas de registro contam segundos observados. VP e FP são linhas com alerta rotuladas como ataque e como normais, respectivamente. Precisão = VP/(VP+FP); recall = VP/(VP+FN); F1 = 2VP/(2VP+FP+FN); taxa de falsos positivos = FP/(FP+VN). As colunas das taxas nos CSVs vão de 0 a 1.

Um episódio é uma sequência de registros consecutivos rotulados como ataque, dentro do mesmo arquivo; uma lacuna na sequência de segundos inicia outro episódio. Basta um alerta dentro dele para contá-lo como sinalizado. A mediana do atraso considera apenas os episódios com alerta; os sem alerta têm atraso ausente. Um episódio sinalizado não implica detectar todas as suas linhas. O agregado externo soma a matriz de confusão dos arquivos antes de calcular as razões; o atraso vem de todos os episódios detectados, sem tirar média das medianas.

## Como abrir ou repetir

1. Extraia o ZIP. Para consultar, abra `07_avaliacao_temporal.ipynb` no VS Code ou consulte os CSVs da pasta `resultados`. O notebook é um leitor dos resultados e traz os comandos de reexecução.
2. Para treinar novamente, use Python 3.12 e instale as versões de `requirements.txt` em um ambiente virtual. Os CSVs brutos não estão no ZIP. Use os originais do [HAI nesta revisão](https://github.com/icsdataset/hai/tree/2a814cebc9a66b06c9e5cd545e2d72e65d383737), pastas `hai-21.03` e `hai-22.04`, preservando a compactação original. São cinco arquivos `test*.csv.gz` na primeira e quatro arquivos `test*.csv` na segunda. Em um clone, baixe também o conteúdo de Git LFS. O script confere os hashes SHA-256 contra `fontes.json`.
3. Dentro da pasta extraída, rode (ajuste os caminhos):

```bash
python comparar_periodos.py --hai21 /caminho/hai-21.03 --hai22 /caminho/hai-22.04 --saida nova_execucao --jobs 3
python resumir.py --saida nova_execucao
python conferir_sinais.py --hai21 /caminho/hai-21.03 --hai22 /caminho/hai-22.04 --saida nova_execucao
```

Para rodar só uma parte, use `--rodadas J1 J2 J3` ou `--rodadas EXT` no primeiro comando. A consolidação exige as quatro rodadas. A conferência dos sinais é uma análise **após** a avaliação; não altera modelos ou previsões.

## Onde encontrar os detalhes

| Arquivo em resultados/ | O que contém |
|---|---|
| metricas_por_periodo.csv | F1, precisão, recall, FP, taxa de FP, cobertura dos episódios e atraso para cada modelo em cada período |
| externo_agregado.csv | Os quatro arquivos do HAI 22.04 somados por modelo |
| episodios_todos.csv | Início, fim, segundos com alerta e atraso de cada episódio |
| limiares_e_validacao.csv | Limiar e F1 de validação; o campo de limiar do ensemble fica vazio por definição |
| protocolo_J1/J2/J3/EXT.json | Listas de sinais, divisões, componentes do PCA, limiares precisos e versões |
| inventario_dos_periodos.csv | Datas reais e quantidades por arquivo de teste |
| diagnostico_faixas_sinais.csv e diagnostico_resumo.csv | Em registros rotulados normais, comparação de cada sinal com o menor e maior valor do treino normal, depois da avaliação |

O pacote experimental original registra scikit-learn 1.9.1; esta análise usa **1.8.0**, com reestimativa dos modelos e limiares. Use os números de J3 para comparar J1/J2/J3 internamente. Preserve os números da execução original na tabela principal do artigo. A verificação não demonstra generalização para outras plantas nem para tipos de ataque garantidamente inéditos.
