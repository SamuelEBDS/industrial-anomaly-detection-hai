# Etapa 4 - Regressão Logística no HAI 21.03

Esta comparação adiciona o segundo classificador clássico ao projeto modular.
Use a pasta do projeto com as etapas 2 e 3 já executadas e a mesma `.venv`.
Extraia o conteúdo deste ZIP na raiz de `tcc-hai_modular`, de modo que
`executar_regressao_logistica.py` fique ao lado de `executar_balanceamento.py`.

No terminal do VS Code, execute:

```bash
./.venv/bin/python executar_regressao_logistica.py --balanceamento balanceamento_rf_20261111_101333
```

Se a sua execução da etapa 3 estiver em outra pasta de `runs/`, passe o nome
dela em `--balanceamento`. O programa confere os dados e grava uma nova pasta
`runs/regressao_logistica_<data_e_hora>/`. Use `--nome NOME_NOVO` para definir
o nome da pasta de saída. `REGRESSÃO LOGÍSTICA CONCLUÍDA.` confirma a execução.

## Protocolo

| Uso dos dados | Arquivos | Operação |
|---|---|---|
| Treino | `test1`, `test2`, `test3` | Define as mesmas 60 entradas, ajusta o `StandardScaler` e treina a Regressão Logística com os 270.003 registros. |
| Validação | `test4` | Calcula pontuações e fixa o limiar pelo maior F1. Em empate, usa o maior limiar. |
| Teste histórico | `test5` | Aplica o modelo e o limiar salvos nas 92.401 linhas originais. |

O modelo usa L2, `C=1`, `lbfgs`, `max_iter=1000` e `tol=1e-6`, sem pesos de
classe e sem subamostragem. A configuração foi fixada antes da avaliação.
A comparação mais direta é com o RF completo sem pesos, treinado com os
mesmos registros. As métricas de RF e IF vêm da execução de balanceamento
indicada no comando; as linhas de validação e teste são conferidas uma a uma.

## Onde olhar

| Arquivo na nova execução | Conteúdo |
|---|---|
| `previsoes_e_metricas/sintese_regressao_logistica.md` | Comparação curta dos modelos e leitura dos intervalos 6 e 9. |
| `previsoes_e_metricas/comparacao_modelos.csv` | Métricas por configuração, inclusive as demais proporções de balanceamento. |
| `previsoes_e_metricas/selecao_validacao.json` | Limiar fixado antes da leitura do teste. |
| `previsoes_e_metricas/{validacao,teste_historico}_metricas.csv` | Precisão, recall, F1, PR-AUC, average precision, matriz de confusão e métricas de episódios da Logística. |
| `previsoes_e_metricas/*_episodios.csv`, `*_falsos_positivos_por_hora.csv` e `*_previsoes.csv.gz` | Detecções por intervalo, tempo até o primeiro alerta, falsos alarmes por hora e pontuações por linha. |
| `previsoes_e_metricas/rotulos_processos_test5.csv` e `intervalos_6_e_9_modelos.csv` | Rótulos por processo e comparação dos modelos nos dois intervalos destacados. |
| `modelo_e_transformacoes/regressao_logistica.joblib` e `manifesto.json` | Pipeline salvo com o limiar e registro da execução. |

Os rótulos `attack_P1`, `attack_P2` e `attack_P3` são lidos separadamente
depois da avaliação para a análise dos intervalos. Eles não fazem parte das
entradas do modelo. O rótulo por processo não identifica sozinho o cenário
de ataque nem a causa dos erros. Como `test5` já era conhecido, a comparação
é histórica e não serve para escolher a configuração.

Para rodar os testes automatizados do projeto com o complemento instalado:

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests
```

A pasta `resultados_preliminares_regressao_logistica/` contém as tabelas de uma execução
com o HAI 21.03. O complemento não inclui os dados nem o ambiente Python.
