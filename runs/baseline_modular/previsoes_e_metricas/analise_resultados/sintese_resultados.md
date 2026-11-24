# Análise de resultados — HAI 21.03

## Protocolo
Treinamento: test1–test3; validação: test4; teste: test5. Modelos e limiares mantidos na etapa de teste.
Os rótulos foram utilizados para selecionar os normais do treinamento dos IFs e para escolher os limiares na validação.

## Teste
92401 registros: 90224 normais e 2177 rotulados como ataque.
A referência que prevê sempre normal alcançaria 97.64% de acurácia e recall zero para ataque.

- RF: precisão 10.95%; recall 60.54%; F1 0.1855; FP 10718; FN 859.
- IF sem PCA: precisão 16.08%; recall 35.00%; F1 0.2203; FP 3978; FN 1415.
- IF com PCA: precisão 52.11%; recall 41.34%; F1 0.4611; FP 827; FN 1277.

Maior F1 neste teste: IF com PCA (0.4611).

## Comparação dos IFs
Com PCA em relação a sem PCA: redução relativa de FP de 79.21%; variação de +138 registros de ataque detectados; variação de recall de +6.34 pontos percentuais.

## Intervalos
12 trechos contínuos identificados pela regra do notebook 4. Intervalos sem alerta em todos os modelos: [6].
- RF: algum alerta em 11 de 12 intervalos.
- IF sem PCA: algum alerta em 11 de 12 intervalos.
- IF com PCA: algum alerta em 10 de 12 intervalos.

## Falsos positivos no tempo
As cinco faixas horárias com mais erros do RF concentram 91.35% dos seus falsos positivos.

## Alcance da análise
O maior F1 agregado não garante presença de alerta em todos os intervalos. Os IDs de intervalo não foram equiparados aos cenários oficiais do HAI. As concentrações de erros e as diferenças entre períodos não estabelecem suas causas. As conclusões se referem às configurações e ao recorte temporal deste experimento.
