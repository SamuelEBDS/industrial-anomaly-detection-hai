"""Limiar, métricas e análises temporais compartilhados pelos notebooks."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score

from . import MODELOS, SUFIXOS

METRICAS_TAXA = ["Acurácia", "Precisão", "Recall", "F1-score",
                 "Taxa de falsos positivos", "Taxa de falsos negativos"]
CONTAGENS = ["Verdadeiros positivos", "Verdadeiros negativos", "Falsos positivos", "Falsos negativos"]
METRICAS = METRICAS_TAXA + CONTAGENS


def divisao_segura(numerador, denominador):
    return numerador / denominador if denominador else 0.0


def _binario(vetor):
    vetor = np.asarray(vetor)
    if vetor.ndim != 1 or not np.isin(vetor, [0, 1]).all():
        raise ValueError("É necessário um vetor de rótulos 0 e 1.")
    return vetor


def calcular_metricas(y_real, y_previsto):
    real, previsto = _binario(y_real), _binario(y_previsto)
    if len(real) == 0 or len(real) != len(previsto):
        raise ValueError("As duas listas precisam ser não vazias e ter o mesmo tamanho.")
    vn, fp, fn, vp = confusion_matrix(real, previsto, labels=[0, 1]).ravel()
    return {
        "Acurácia": accuracy_score(real, previsto),
        "Precisão": precision_score(real, previsto, zero_division=0),
        "Recall": recall_score(real, previsto, zero_division=0),
        "F1-score": f1_score(real, previsto, zero_division=0),
        "Verdadeiros positivos": int(vp), "Verdadeiros negativos": int(vn),
        "Falsos positivos": int(fp), "Falsos negativos": int(fn),
        "Taxa de falsos positivos": divisao_segura(fp, fp + vn),
        "Taxa de falsos negativos": divisao_segura(fn, fn + vp),
    }


def selecionar_limiar(y_real, pontuacoes):
    real = _binario(y_real)
    scores = np.asarray(pontuacoes)
    if len(real) != len(scores) or scores.ndim != 1 or not np.isfinite(scores).all() or len(np.unique(real)) != 2:
        raise ValueError("A calibração precisa de duas classes e de uma pontuação finita por registro.")
    precisao, recall, limiares = precision_recall_curve(real, scores, pos_label=1)
    precisao, recall = precisao[:-1], recall[:-1]
    denominador = precisao + recall
    valores_f1 = np.divide(2 * precisao * recall, denominador,
                          out=np.zeros_like(denominador), where=denominador > 0)
    indice = np.flatnonzero(valores_f1 == valores_f1.max())[-1]
    return float(limiares[indice])


def aplicar_limiar(pontuacoes, limiar):
    scores = np.asarray(pontuacoes)
    if not np.isfinite(scores).all() or not np.isfinite(limiar):
        raise ValueError("Pontuações e limiar precisam ser finitos.")
    return (scores >= limiar).astype(int)


def resumir_previsoes(y_real, previsoes):
    return pd.DataFrame({nome: calcular_metricas(y_real, p) for nome, p in previsoes.items()})


def montar_resultados(metadados, y, pontuacoes, previsoes, limiares):
    if len(metadados) != len(y) or not metadados.index.equals(y.index):
        raise ValueError("Metadados e rótulos desalinhados.")
    resultado = metadados.copy()
    resultado["attack_real"] = y.to_numpy()
    for nome, sufixo in SUFIXOS.items():
        resultado[f"pontuacao_{sufixo}"] = pontuacoes[nome]
        resultado[f"previsao_{sufixo}"] = previsoes[nome]
        resultado[f"limiar_{sufixo}"] = limiares[nome]
    return resultado


def identificar_intervalos(resultados):
    analise = resultados.sort_values(["arquivo_origem", "time"], kind="stable").reset_index(drop=True).copy()
    em_ataque = analise["attack_real"].eq(1)
    inicio = em_ataque & (
        ~em_ataque.shift(fill_value=False)
        | analise["time"].diff().ne(pd.Timedelta(seconds=1))
        | analise["arquivo_origem"].ne(analise["arquivo_origem"].shift())
    )
    analise["intervalo_id"] = inicio.cumsum().where(em_ataque, 0).astype(int)
    resumo = analise.loc[em_ataque].groupby("intervalo_id").agg(
        inicio=("time", "min"), fim=("time", "max"),
        registros_ataque=("attack_real", "size"),
    ).reset_index()
    return analise, resumo


def avaliar_intervalos(analise):
    avaliacoes = []
    for identificador, trecho in analise.loc[analise["intervalo_id"] > 0].groupby("intervalo_id"):
        inicio, total = trecho["time"].min(), len(trecho)
        for nome, sufixo in SUFIXOS.items():
            alertas = trecho[f"previsao_{sufixo}"].eq(1)
            detectados = int(alertas.sum())
            atraso = (trecho.loc[alertas, "time"].min() - inicio).total_seconds() if detectados else np.nan
            avaliacoes.append({
                "intervalo_id": int(identificador), "modelo": nome, "registros_ataque": total,
                "registros_detectados": detectados, "registros_nao_detectados": total - detectados,
                "cobertura_percentual": 100 * detectados / total,
                "primeiro_alerta_segundos": atraso,
            })
    colunas = ["intervalo_id", "modelo", "registros_ataque", "registros_detectados",
               "registros_nao_detectados", "cobertura_percentual", "primeiro_alerta_segundos"]
    detalhes = pd.DataFrame(avaliacoes, columns=colunas)
    cobertura = detalhes.pivot(index="intervalo_id", columns="modelo", values="cobertura_percentual").reindex(columns=MODELOS)
    atraso = detalhes.pivot(index="intervalo_id", columns="modelo", values="primeiro_alerta_segundos").reindex(columns=MODELOS)
    return detalhes, cobertura, atraso


def falsos_positivos_por_hora(resultados):
    normais = resultados.loc[resultados["attack_real"].eq(0)].copy()
    normais["hora"] = normais["time"].dt.floor("h")
    grupos = normais.groupby("hora")
    tabela = grupos.size().to_frame("Registros normais")
    for nome, sufixo in SUFIXOS.items():
        tabela[f"FP {nome}"] = grupos[f"previsao_{sufixo}"].sum()
        tabela[f"Taxa FP {nome} (%)"] = 100 * tabela[f"FP {nome}"] / tabela["Registros normais"]
    return tabela


def carregar_metricas(caminho):
    tabela = pd.read_csv(caminho, index_col="Metrica")
    if not set(MODELOS).issubset(tabela.columns) or not set(METRICAS).issubset(tabela.index):
        raise ValueError(f"Estrutura de métricas inesperada: {caminho}")
    tabela = tabela.loc[METRICAS, list(MODELOS)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(tabela.to_numpy()).all():
        raise ValueError("Métrica não finita.")
    for modelo in MODELOS:
        valores = tabela.loc[CONTAGENS, modelo].to_numpy()
        if (valores < 0).any() or not np.allclose(valores, np.round(valores)):
            raise ValueError("Contagens inválidas.")
        vp, vn, fp, fn = np.round(valores).astype(int)
        precisao, recall = divisao_segura(vp, vp + fp), divisao_segura(vp, vp + fn)
        esperadas = [divisao_segura(vp + vn, vp + vn + fp + fn), precisao, recall,
                     divisao_segura(2 * precisao * recall, precisao + recall),
                     divisao_segura(fp, fp + vn), divisao_segura(fn, fn + vp)]
        if not np.allclose(tabela.loc[METRICAS_TAXA, modelo], esperadas, atol=1e-4, rtol=0):
            raise ValueError("Taxas incompatíveis com as contagens.")
    normais = tabela.loc["Verdadeiros negativos"] + tabela.loc["Falsos positivos"]
    ataques = tabela.loc["Verdadeiros positivos"] + tabela.loc["Falsos negativos"]
    if normais.nunique() != 1 or ataques.nunique() != 1:
        raise ValueError("Modelos avaliados em conjuntos diferentes.")
    return tabela
