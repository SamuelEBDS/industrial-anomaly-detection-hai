"""Métricas e resumos da comparação de balanceamento."""
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from ..evaluation import aplicar_limiar, calcular_metricas, identificar_intervalos


def preparar_avaliacao(metadados, y):
    if not metadados.index.equals(y.index):
        raise ValueError("Rótulos e metadados precisam estar alinhados.")
    base = metadados.assign(attack_real=y.to_numpy(), posicao=np.arange(len(y)))
    com_id, intervalos = identificar_intervalos(base)
    base["intervalo_id"] = com_id.sort_values("posicao")["intervalo_id"].to_numpy()
    return base.drop(columns="posicao"), intervalos


def avaliar(base, intervalos, scores, limiar):
    previsto = aplicar_limiar(scores, limiar)
    real = base["attack_real"]
    m = calcular_metricas(real, previsto)
    precisao, recall, _ = precision_recall_curve(real, scores, pos_label=1)
    resultados = base.assign(pontuacao=scores, alerta=previsto)
    normais = resultados.loc[real.eq(0)].assign(hora=lambda d: d["time"].dt.floor("h"))
    horas = normais.groupby(["arquivo_origem", "hora"], as_index=False).agg(
        registros_normais=("alerta", "size"), falsos_positivos=("alerta", "sum"),
        pontuacao_mediana=("pontuacao", "median"), pontuacao_p90=("pontuacao", lambda s: s.quantile(.9)),
    )
    horas["taxa_fp"] = horas["falsos_positivos"] / horas["registros_normais"]

    alertas = resultados.loc[real.eq(1) & resultados["alerta"].eq(1)]
    detectados = alertas.groupby("intervalo_id", as_index=False).agg(
        registros_detectados=("alerta", "size"), primeiro_alerta=("time", "min"),
    )
    episodios = intervalos.merge(detectados, on="intervalo_id", how="left", validate="one_to_one")
    episodios["registros_detectados"] = episodios["registros_detectados"].fillna(0).astype(int)
    episodios["cobertura"] = episodios["registros_detectados"] / episodios["registros_ataque"]
    episodios["atraso_segundos"] = (episodios["primeiro_alerta"] - episodios["inicio"]).dt.total_seconds()
    atraso = episodios["atraso_segundos"].dropna()
    metricas = {
        "limiar": limiar, "precisao": m["Precisão"], "recall": m["Recall"], "f1": m["F1-score"],
        "pr_auc": float(auc(recall, precisao)), "average_precision": float(average_precision_score(real, scores)),
        "acuracia": m["Acurácia"], "verdadeiros_positivos": m["Verdadeiros positivos"],
        "verdadeiros_negativos": m["Verdadeiros negativos"], "falsos_positivos": m["Falsos positivos"],
        "falsos_negativos": m["Falsos negativos"], "taxa_fp": m["Taxa de falsos positivos"],
        "episodios_detectados": int(episodios["primeiro_alerta"].notna().sum()),
        "episodios_totais": len(episodios), "atraso_mediano_segundos": atraso.median() if len(atraso) else np.nan,
    }
    return metricas, horas, episodios, previsto


def resumir(metricas):
    colunas = ["f1", "precisao", "recall", "pr_auc", "average_precision", "falsos_positivos", "taxa_fp",
               "episodios_detectados", "atraso_mediano_segundos"]
    agregacoes = {f"{c}_media": (c, "mean") for c in colunas}
    return metricas.groupby(["particao", "configuracao"], as_index=False, sort=False).agg(
        execucoes=("variante", "size"), f1_dp=("f1", "std"), f1_min=("f1", "min"), f1_max=("f1", "max"),
        episodios_totais=("episodios_totais", "first"), **agregacoes,
    )


def ordenar_rf_validacao(resumo_validacao):
    if set(resumo_validacao["particao"]) != {"validacao"}:
        raise ValueError("A escolha de configuração só pode usar a validação.")
    rfs = resumo_validacao.loc[resumo_validacao["configuracao"].str.startswith("RF ")].copy()
    return rfs.sort_values(["f1_media", "falsos_positivos_media", "configuracao"],
                           ascending=[False, True, True]).reset_index(drop=True)


def investigar_fp(horas_teste, base_teste, X_teste, base_validacao, X_validacao):
    rf = horas_teste.loc[horas_teste["variante"].eq("rf_ponderado")]
    top = rf.sort_values(["falsos_positivos", "hora"], ascending=[False, True]).head(5).copy()
    for variante in ("if_sem_pca", "if_com_pca"):
        alternativa = horas_teste.loc[horas_teste["variante"].eq(variante), ["hora", "falsos_positivos"]]
        top = top.merge(alternativa.rename(columns={"falsos_positivos": f"fp_{variante}"}),
                        on="hora", validate="one_to_one")

    normal_teste = base_teste["attack_real"].eq(0)
    hora_top = base_teste["time"].dt.floor("h").isin(top["hora"])
    referencia = X_validacao.loc[base_validacao["attack_real"].eq(0)]
    quantis = referencia.quantile([.25, .75])
    iqr = quantis.loc[.75] - quantis.loc[.25]
    altas = X_teste.loc[normal_teste & hora_top].median()
    restantes = X_teste.loc[normal_teste & ~hora_top].median()
    sinais = pd.DataFrame({
        "mediana_normais_validacao": referencia.median(), "iqr_normais_validacao": iqr,
        "mediana_normais_top5": altas, "mediana_normais_outras_horas": restantes,
        "diferenca_em_iqr": (altas - restantes).abs() / iqr.where(iqr.gt(0)),
    }).rename_axis("sinal").reset_index().sort_values("diferenca_em_iqr", ascending=False)
    return top, sinais
