# HAI 21.03 — comparação de balanceamento do Random Forest

Os novos RFs usam 60 sinais, 100 árvores e os mesmos parâmetros, com pesos de classe desligados. Foram mantidos os 5.613 registros rotulados como ataque; são segundos correlacionados. Só a quantidade e a seleção dos registros normais de treino variam entre as proporções. A referência ponderada e os Isolation Forest são os modelos já conferidos na etapa modular.

**Escolha em `test4`: RF sem pesos. Melhor configuração subamostrada: RF 1:20.** A escolha foi registrada antes de avaliar `test5`. As duas partições mantiveram as distribuições originais. Cada modelo usa seu próprio corte escolhido pelo F1 em `test4`.

| Configuração | Rodadas | F1 médio `test4` | DP em `test4` | F1 médio `test5` | DP em `test5` | Precisão `test5` | Recall `test5` | PR-AUC `test5` | FP médios `test5` | Episódios médios `test5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF ponderado | 1 | 0,6414 | — | 0,1855 | — | 10,95% | 60,54% | 0,2077 | 10.718 | 11/12 |
| IF sem PCA | 1 | 0,2733 | — | 0,2203 | — | 16,08% | 35,00% | 0,1636 | 3.978 | 11/12 |
| IF com PCA | 1 | 0,6429 | — | 0,4611 | — | 52,11% | 41,34% | 0,3801 | 827 | 10/12 |
| RF sem pesos | 1 | 0,8485 | — | 0,1804 | — | 10,27% | 73,86% | 0,2651 | 14.042 | 12/12 |
| RF 1:20 | 3 | 0,7093 | 0,0097 | 0,1781 | 0,0016 | 10,21% | 69,85% | 0,2878 | 13.379,0 | 12,0/12 |
| RF 1:10 | 3 | 0,6901 | 0,0254 | 0,1736 | 0,0053 | 10,05% | 63,56% | 0,2391 | 12.390,0 | 12,0/12 |
| RF 1:5 | 3 | 0,6791 | 0,0437 | 0,1919 | 0,0235 | 11,73% | 56,18% | 0,2867 | 9.565,0 | 11,3/12 |
| RF 1:1 | 3 | 0,6780 | 0,0286 | 0,1796 | 0,0135 | 10,47% | 63,45% | 0,3321 | 11.862,0 | 11,3/12 |

DP: desvio padrão amostral entre as três sementes de amostragem. Para as referências e o RF completo sem pesos, há só uma rodada. PR-AUC é a área trapezoidal da curva precisão–recall; o CSV também contém average precision. As médias resumem modelos treinados separadamente e avaliados nos mesmos períodos.

O RF 1:20, escolhido entre as proporções em `test4`, obteve F1 médio de 0,7093 na validação e 0,1781 no teste histórico. O RF completo sem pesos obteve 0,8485 e 0,1804, respectivamente. Esse é o controle para comparar o efeito de reduzir os normais sem misturá-lo à retirada dos pesos de classe.

No teste histórico, o RF 1:20 teve recall médio de 69,85%, com 13.379,0 falsos positivos em média. O RF completo sem pesos teve recall de 73,86% e 14.042 falsos positivos. Para julgar a subamostragem, leia as duas medidas junto com o F1 e a variação entre sementes.

## Por que o F1 do RF ponderado caiu?

Em `test4`, o RF ponderado teve F1 0,6414, recall 61,45% e 349 falsos positivos (0,91% dos normais). Em `test5`, o F1 foi 0,1855, o recall 60,54% e foram 10.718 falsos positivos (11,88% dos normais). A precisão caiu de 67,08% para 10,95%. As cinco horas abaixo concentram 91,35% dos falsos positivos do RF ponderado em `test5`.

| Início da hora | Normais | FP do RF ponderado | Taxa FP do RF | FP do IF sem PCA | FP do IF com PCA |
|---|---:|---:|---:|---:|---:|
| 31/07 09:00 | 3.338 | 3.166 | 94,85% | 1.129 | 126 |
| 31/07 10:00 | 3.480 | 2.951 | 84,80% | 656 | 54 |
| 31/07 11:00 | 3.338 | 1.479 | 44,31% | 309 | 231 |
| 31/07 08:00 | 3.600 | 1.372 | 38,11% | 611 | 109 |
| 31/07 12:00 | 1.801 | 823 | 45,70% | 294 | 6 |

A elevação dos falsos positivos, junto ao recall próximo, explica a queda da precisão e do F1. A causa física dos alertas exige mais evidência. Entre os 60 sinais, P1_FCV03Z, P1_B3005, P1_FCV03D tiveram os maiores deslocamentos de mediana entre essas cinco horas e as demais, escalados pelo intervalo interquartil dos normais de `test4`. Isso indica sinais para investigar, sem estabelecer causalidade ou importância para o modelo.

## Intervalos 6 e 9 de `test5`

Consideramos um episódio detectado quando há pelo menos um alerta dentro do trecho rotulado. O atraso é o tempo até esse primeiro alerta. Um traço indica episódio sem alerta.

| Intervalo | Configuração | Semente da amostra | Registros detectados | Primeiro alerta (s) |
|---|---|---:|---:|---:|
| 6 | IF com PCA | — | 0/153 | — |
| 6 | IF sem PCA | — | 0/153 | — |
| 6 | RF 1:20 | 13 | 58/153 | 7 |
| 6 | RF 1:20 | 42 | 59/153 | 7 |
| 6 | RF 1:20 | 2026 | 60/153 | 4 |
| 6 | RF ponderado | — | 0/153 | — |
| 6 | RF sem pesos | — | 62/153 | 4 |
| 9 | IF com PCA | — | 0/241 | — |
| 9 | IF sem PCA | — | 14/241 | 72 |
| 9 | RF 1:20 | 13 | 31/241 | 70 |
| 9 | RF 1:20 | 42 | 55/241 | 70 |
| 9 | RF 1:20 | 2026 | 41/241 | 70 |
| 9 | RF ponderado | — | 71/241 | 70 |
| 9 | RF sem pesos | — | 75/241 | 69 |

Os IDs são trechos contínuos do rótulo `attack`. A associação de cada intervalo a um cenário oficial de ataque não foi verificada. `test5` já era conhecido; uma nova janela reservada é necessária para confirmar a generalização.
