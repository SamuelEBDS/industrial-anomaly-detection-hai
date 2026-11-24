"""Apresenta as métricas da Logística junto com as referências já avaliadas."""
import pandas as pd


def _n(valor, casas=0):
    return '—' if pd.isna(valor) else f'{valor:,.{casas}f}'.translate(str.maketrans(',.', '.,'))


def _pct(taxa):
    return f'{_n(100 * taxa, 2)}%'


def gerar_sintese(saida, comparacao, teste_logistica, rotulos, episodios_logistica, logistica, melhor_subamostrado):
    val = comparacao.loc[comparacao["particao"].eq("validacao")].set_index("configuracao")
    teste = comparacao.loc[comparacao["particao"].eq("teste_historico")].set_index("configuracao")
    modelos = ["RF ponderado", "RF sem pesos", melhor_subamostrado, "IF sem PCA", "IF com PCA", logistica]
    linhas = [
        "# HAI 21.03: Regressão Logística",
        "",
        "A Regressão Logística usa os mesmos 60 sinais e todos os 270.003 registros de `test1` a `test3`, "
        "sem pesos de classe nem subamostragem. O `StandardScaler` aprende média e escala apenas nesse treino. "
        "O ajuste usa regularização L2, `C=1` e o solver `lbfgs`.",
        "",
        f"**Limiar fixado em `test4`: {_n(teste_logistica['limiar'], 8)}**, pelo maior F1. "
        "Em `test5`, as 92.401 linhas e a distribuição original foram preservadas. "
        "Os resultados desse período são uma comparação histórica.",
        "",
        "| Configuração | F1 `test4` | F1 `test5` | Precisão `test5` | Recall `test5` | PR-AUC `test5` | FP `test5` | Episódios `test5` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nome in modelos:
        v, t = val.loc[nome], teste.loc[nome]
        casas = 1 if t["execucoes"] > 1 else 0
        linhas.append(f"| {nome} | {_n(v['f1_media'],4)} | {_n(t['f1_media'],4)} | {_pct(t['precisao_media'])} | "
                      f"{_pct(t['recall_media'])} | {_n(t['pr_auc_media'],4)} | {_n(t['falsos_positivos_media'],casas)} | "
                      f"{_n(t['episodios_detectados_media'],casas)}/{_n(t['episodios_totais'])} |")
    linhas += [
        "",
        f"A configuração subamostrada escolhida em `test4` foi {melhor_subamostrado}; sua linha usa médias de três sementes. "
        "`comparacao_modelos.csv` inclui as demais proporções. PR-AUC é a área trapezoidal da curva precisão–recall; "
        "average precision e contagens da matriz de confusão estão nos CSVs de métricas.",
        "",
        f"No teste histórico, a Logística marcou {_n(teste_logistica['verdadeiros_positivos'])} registros de ataque "
        f"e {_n(teste_logistica['falsos_positivos'])} registros normais. "
        f"Deixou passar {_n(teste_logistica['falsos_negativos'])} registros rotulados como ataque. "
        f"Teve alerta em {_n(teste_logistica['episodios_detectados'])} de {_n(teste_logistica['episodios_totais'])} episódios. "
        f"O atraso mediano entre os episódios detectados foi de {_n(teste_logistica['atraso_mediano_segundos'],1)} s." if teste_logistica['episodios_detectados'] else
        f"No teste histórico, a Logística teve {_n(teste_logistica['falsos_positivos'])} falsos positivos e não detectou episódios.",
        "",
        "## Intervalos 6 e 9: rótulos de processo",
        "",
        "| Intervalo | Registros de ataque | Marcados P1 | Marcados P2 | Marcados P3 | Detectados pela Logística | Primeiro alerta (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    detalhes = rotulos.merge(episodios_logistica[["intervalo_id", "registros_detectados", "atraso_segundos"]],
                             on="intervalo_id", validate="one_to_one")
    for r in detalhes.loc[detalhes["intervalo_id"].isin([6,9])].itertuples():
        linhas.append(f"| {r.intervalo_id} | {_n(r.registros_ataque)} | {_n(r.attack_P1)} | {_n(r.attack_P2)} | "
                      f"{_n(r.attack_P3)} | {_n(r.registros_detectados)} | {_n(r.atraso_segundos)} |")
    linhas += [
        "",
        "Um episódio é detectado se houver pelo menos um alerta dentro do intervalo rotulado. Um traço indica ausência de alerta. "
        "Os rótulos P1, P2 e P3 foram usados apenas nesta análise; o mesmo registro pode ter mais de uma marcação. "
        "O cruzamento identifica os processos marcados, sem estabelecer o cenário oficial ou a causa física dos erros.",
        "",
    ]
    (saida / "sintese_regressao_logistica.md").write_text("\n".join(linhas), encoding="utf-8")
