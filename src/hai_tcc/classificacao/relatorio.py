"""Síntese descritiva dos resultados; a seleção já está fixada em test4."""
import pandas as pd


def _n(valor, casas=0):
    if pd.isna(valor):
        return "—"
    return f"{valor:,.{casas}f}".translate(str.maketrans(",.", ".,"))


def _pct(taxa):
    return _n(100 * taxa, 2) + "%"


def gerar_relatorio(destino, selecao, resumo, validacao, teste, top5, sinais, intervalos):
    val = resumo.loc[resumo["particao"].eq("validacao")].set_index("configuracao")
    hist = resumo.loc[resumo["particao"].eq("teste_historico")].set_index("configuracao")
    melhor, amostrado = selecao["melhor_rf"], selecao["melhor_rf_subamostrado"]
    hist_amostra, hist_sem_pesos = hist.loc[amostrado], hist.loc["RF sem pesos"]
    ref_v = validacao.set_index("variante").loc["rf_ponderado"]
    ref_t = teste.set_index("variante").loc["rf_ponderado"]
    concentracao = top5["falsos_positivos"].sum() / ref_t["falsos_positivos"]
    linhas = [
        "# HAI 21.03 — comparação de balanceamento do Random Forest",
        "",
        "Os novos RFs usam 60 sinais, 100 árvores e os mesmos parâmetros, com pesos de classe desligados. "
        "Foram mantidos os 5.613 registros rotulados como ataque; são segundos correlacionados. "
        "Só a quantidade e a seleção dos registros normais de treino variam entre as proporções. "
        "A referência ponderada e os Isolation Forest são os modelos já conferidos na etapa modular.",
        "",
        f"**Escolha em `test4`: {melhor}. Melhor configuração subamostrada: {amostrado}.** "
        "A escolha foi registrada antes de avaliar `test5`. As duas partições mantiveram as distribuições originais. "
        "Cada modelo usa seu próprio corte escolhido pelo F1 em `test4`.",
        "",
        "| Configuração | Rodadas | F1 médio `test4` | DP em `test4` | F1 médio `test5` | DP em `test5` | Precisão `test5` | Recall `test5` | PR-AUC `test5` | FP médios `test5` | Episódios médios `test5` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config, v in val.iterrows():
        t = hist.loc[config]
        dec = 1 if t["execucoes"] > 1 else 0
        linhas.append(f"| {config} | {_n(t['execucoes'])} | {_n(v['f1_media'],4)} | {_n(v['f1_dp'],4)} | "
                      f"{_n(t['f1_media'],4)} | {_n(t['f1_dp'],4)} | {_pct(t['precisao_media'])} | "
                      f"{_pct(t['recall_media'])} | {_n(t['pr_auc_media'],4)} | {_n(t['falsos_positivos_media'],dec)} | "
                      f"{_n(t['episodios_detectados_media'],dec)}/{_n(t['episodios_totais'])} |")
    linhas += [
        "",
        "DP: desvio padrão amostral entre as três sementes de amostragem. Para as referências e o RF completo "
        "sem pesos, há só uma rodada. PR-AUC é a área trapezoidal da curva precisão–recall; o CSV também "
        "contém average precision. As médias resumem modelos treinados separadamente e avaliados nos mesmos períodos.",
        "",
        f"O {amostrado}, escolhido entre as proporções em `test4`, obteve F1 médio de {_n(val.loc[amostrado, 'f1_media'],4)} "
        f"na validação e {_n(hist.loc[amostrado, 'f1_media'],4)} no teste histórico. "
        f"O RF completo sem pesos obteve {_n(val.loc['RF sem pesos','f1_media'],4)} e {_n(hist.loc['RF sem pesos','f1_media'],4)}, respectivamente. "
        "Esse é o controle para comparar o efeito de reduzir os normais sem misturá-lo à retirada dos pesos de classe.",
        "",
        f"No teste histórico, o {amostrado} teve recall médio de {_pct(hist_amostra['recall_media'])}, "
        f"com {_n(hist_amostra['falsos_positivos_media'],1)} falsos positivos em média. "
        f"O RF completo sem pesos teve recall de {_pct(hist_sem_pesos['recall_media'])} "
        f"e {_n(hist_sem_pesos['falsos_positivos_media'])} falsos positivos. "
        "Para julgar a subamostragem, leia as duas medidas junto com o F1 e a variação entre sementes.",
        "",
        "## Por que o F1 do RF ponderado caiu?",
        "",
        f"Em `test4`, o RF ponderado teve F1 {_n(ref_v['f1'],4)}, recall {_pct(ref_v['recall'])} e "
        f"{_n(ref_v['falsos_positivos'])} falsos positivos ({_pct(ref_v['taxa_fp'])} dos normais). "
        f"Em `test5`, o F1 foi {_n(ref_t['f1'],4)}, o recall {_pct(ref_t['recall'])} e foram "
        f"{_n(ref_t['falsos_positivos'])} falsos positivos ({_pct(ref_t['taxa_fp'])} dos normais). "
        f"A precisão caiu de {_pct(ref_v['precisao'])} para {_pct(ref_t['precisao'])}. "
        f"As cinco horas abaixo concentram {_pct(concentracao)} dos falsos positivos do RF ponderado em `test5`.",
        "",
        "| Início da hora | Normais | FP do RF ponderado | Taxa FP do RF | FP do IF sem PCA | FP do IF com PCA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in top5.itertuples():
        linhas.append(f"| {r.hora:%d/%m %H:%M} | {_n(r.registros_normais)} | {_n(r.falsos_positivos)} | "
                      f"{_pct(r.taxa_fp)} | {_n(r.fp_if_sem_pca)} | {_n(r.fp_if_com_pca)} |")
    candidatos = sinais.dropna(subset=["diferenca_em_iqr"]).head(3)["sinal"].tolist()
    linhas += [
        "",
        "A elevação dos falsos positivos, junto ao recall próximo, explica a queda da precisão e do F1. "
        "A causa física dos alertas exige mais evidência. "
        f"Entre os 60 sinais, {', '.join(candidatos)} tiveram os maiores deslocamentos de mediana entre essas cinco horas "
        "e as demais, escalados pelo intervalo interquartil dos normais de `test4`. "
        "Isso indica sinais para investigar, sem estabelecer causalidade ou importância para o modelo.",
        "",
        "## Intervalos 6 e 9 de `test5`",
        "",
        "Consideramos um episódio detectado quando há pelo menos um alerta dentro do trecho rotulado. "
        "O atraso é o tempo até esse primeiro alerta. Um traço indica episódio sem alerta.",
        "",
        "| Intervalo | Configuração | Semente da amostra | Registros detectados | Primeiro alerta (s) |",
        "|---|---|---:|---:|---:|",
    ]
    comparados = intervalos.loc[intervalos["configuracao"].isin(["RF ponderado", "RF sem pesos", "IF sem PCA", "IF com PCA", amostrado])]
    for r in comparados.sort_values(["intervalo_id", "configuracao", "semente_amostragem"]).itertuples():
        semente = "—" if pd.isna(r.semente_amostragem) else str(r.semente_amostragem)
        linhas.append(f"| {r.intervalo_id} | {r.configuracao} | {semente} | "
                      f"{_n(r.registros_detectados)}/{_n(r.registros_ataque)} | {_n(r.atraso_segundos)} |")
    linhas += [
        "",
        "Os IDs são trechos contínuos do rótulo `attack`. A associação de cada intervalo a um cenário oficial "
        "de ataque não foi verificada. `test5` já era conhecido; uma nova janela reservada é necessária para "
        "confirmar a generalização.",
        "",
    ]
    (destino / "sintese_balanceamento.md").write_text("\n".join(linhas), encoding="utf-8")
