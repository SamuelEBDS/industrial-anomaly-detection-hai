# Etapa 3 — comparação de balanceamento no HAI 21.03

**Pergunta:** manter os ataques e reduzir os registros normais do treino ajuda o Random Forest a detectar ataques com menos alarmes falsos?

Este complemento usa a versão modular conferida. Extraia o conteúdo do ZIP na raiz de `tcc-hai_modular`, a mesma pasta de `executar.py`. O novo `executar_balanceamento.py` fica ao lado dele. Cada execução cria sua própria pasta `runs/balanceamento_rf_<data_e_hora>/`.

No terminal do VS Code, abra a pasta do projeto e execute:

```bash
./.venv/bin/python executar_balanceamento.py
```

Use o mesmo Python dos cinco notebooks, com os dados e os resultados de `runs/baseline_modular` já disponíveis. Se a reprodução modular foi salva com outro nome, use `--referencia NOME_DA_EXECUCAO`. Para escolher um nome de saída, use `--nome NOME_NOVO`.

## O que será comparado

| Configuração de RF | Registros normais de treino | Registros de ataque | Rodadas |
|---|---:|---:|---:|
| RF ponderado (referência) | 264.390 | 5.613 | 1 |
| RF completo sem pesos | 264.390 | 5.613 | 1 |
| 1 ataque para 20 normais | 112.260 | 5.613 | 3 |
| 1 ataque para 10 normais | 56.130 | 5.613 | 3 |
| 1 ataque para 5 normais | 28.065 | 5.613 | 3 |
| 1 ataque para 1 normal | 5.613 | 5.613 | 3 |

**Macete:** pense em uma pilha de cartões de treinamento. Todos os 5.613 cartões de ataque ficam. A proporção 1:5 separa cinco cartões normais para cada cartão de ataque. O sorteio de normais respeita a distribuição por arquivo e hora, sem reposição. As sementes 13, 42 e 2026 repetem esse sorteio. A semente interna do Random Forest fica fixa, assim como as 100 árvores e a ordem das 60 entradas. Os RFs com subamostragem não usam pesos de classe.

## Como ler os resultados

1. O treino usa `test1`, `test2` e `test3`. Apenas o treino dos RFs 1:20, 1:10, 1:5 e 1:1 é subamostrado.
2. Cada limiar é escolhido pelo maior F1 em `test4`, com todas as suas linhas. O ranking escolhe a configuração pelo **F1 médio** das rodadas; em empate, prefere menos falsos positivos médios. A escolha e os limiares ficam registrados antes de ler `test5`.
3. Em `test5`, o programa aplica os cortes fixados e mede o resultado histórico com as 92.401 linhas originais. RF ponderado, IF sem PCA e IF com PCA entram com as previsões já conferidas. A escolha continua sendo a de `test4`, ainda que outra proporção pareça melhor em `test5`.

Quando aparecer `BALANCEAMENTO CONCLUÍDO.`, abra o arquivo **`sintese_balanceamento.md`** no caminho indicado pelo terminal. Para examinar os números ou repetir a execução, estes são os arquivos principais:

| Arquivo ou pasta | Conteúdo |
|---|---|
| `previsoes_e_metricas/ranking_rf_validacao.csv` | Qual configuração ficou na frente em `test4`. |
| `previsoes_e_metricas/selecao_validacao.json` | Escolhas e cortes fixados antes de `test5`. |
| `previsoes_e_metricas/comparacao_configuracoes.csv` | Médias, desvio padrão do F1, precisão, recall, PR-AUC, FP e cobertura dos episódios. |
| `previsoes_e_metricas/{validacao,teste_historico}_metricas.csv` | Métricas e corte de cada rodada. A área PR trapezoidal e a average precision são registradas separadamente. |
| `previsoes_e_metricas/*_episodios.csv` e `*_falsos_positivos_por_hora.csv` | Registros detectados, tempo até o primeiro alerta e falsos alarmes por hora. |
| `previsoes_e_metricas/cinco_horas_mais_fp_rf_ponderado.csv` e `intervalos_6_e_9.csv` | Evidências para discutir as falhas da referência. |
| `amostras/` e `modelo_e_transformacoes/` | Índices de origem das amostras, modelos novos e cortes. |
| `manifesto.json` | Revisão dos dados, versões, configuração, sementes e horários das etapas. |

`resultados_balanceamento/` contém tabelas e a síntese de uma execução com os dados reais do HAI e Python 3.14.4, para comparar com a sua. Para compartilhar, o código e as tabelas bastam; os notebooks e o conjunto de dados continuam na versão modular.

Plano de referência: *Orientações para a próxima etapa — TCC / HAI 21.03*, p. 2.
