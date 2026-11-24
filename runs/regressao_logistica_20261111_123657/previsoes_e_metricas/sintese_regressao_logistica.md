# HAI 21.03: Regressão Logística

A Regressão Logística usa os mesmos 60 sinais e todos os 270.003 registros de `test1` a `test3`, sem pesos de classe nem subamostragem. O `StandardScaler` aprende média e escala apenas nesse treino. O ajuste usa regularização L2, `C=1` e o solver `lbfgs`.

**Limiar fixado em `test4`: 0,15296581**, pelo maior F1. Em `test5`, as 92.401 linhas e a distribuição original foram preservadas. Os resultados desse período são uma comparação histórica.

| Configuração | F1 `test4` | F1 `test5` | Precisão `test5` | Recall `test5` | PR-AUC `test5` | FP `test5` | Episódios `test5` |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF ponderado | 0,6414 | 0,1855 | 10,95% | 60,54% | 0,2077 | 10.718 | 11/12 |
| RF sem pesos | 0,8485 | 0,1804 | 10,27% | 73,86% | 0,2651 | 14.042 | 12/12 |
| RF 1:20 | 0,7093 | 0,1781 | 10,21% | 69,85% | 0,2878 | 13.379,0 | 12,0/12 |
| IF sem PCA | 0,2733 | 0,2203 | 16,08% | 35,00% | 0,1636 | 3.978 | 11/12 |
| IF com PCA | 0,6429 | 0,4611 | 52,11% | 41,34% | 0,3801 | 827 | 10/12 |
| Regressão Logística sem pesos | 0,7779 | 0,5671 | 63,45% | 51,26% | 0,5252 | 643 | 12/12 |

A configuração subamostrada escolhida em `test4` foi RF 1:20; sua linha usa médias de três sementes. `comparacao_modelos.csv` inclui as demais proporções. PR-AUC é a área trapezoidal da curva precisão–recall; average precision e contagens da matriz de confusão estão nos CSVs de métricas.

No teste histórico, a Logística marcou 1.116 registros de ataque e 643 registros normais. Deixou passar 1.061 registros rotulados como ataque. Teve alerta em 12 de 12 episódios. O atraso mediano entre os episódios detectados foi de 21,0 s.

## Intervalos 6 e 9: rótulos de processo

| Intervalo | Registros de ataque | Marcados P1 | Marcados P2 | Marcados P3 | Detectados pela Logística | Primeiro alerta (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 153 | 0 | 152 | 120 | 118 | 0 |
| 9 | 241 | 241 | 0 | 0 | 23 | 94 |

Um episódio é detectado se houver pelo menos um alerta dentro do intervalo rotulado. Um traço indica ausência de alerta. Os rótulos P1, P2 e P3 foram usados apenas nesta análise; o mesmo registro pode ter mais de uma marcação. O cruzamento identifica os processos marcados, sem estabelecer o cenário oficial ou a causa física dos erros.
