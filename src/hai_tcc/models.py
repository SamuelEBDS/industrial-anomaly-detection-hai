"""Treino e pontuações dos modelos; pontuar nunca reajusta transformações."""
from time import perf_counter

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from .features import validar_entradas


def empacotar(modelo, colunas, constantes, arquivos, tempo, **extras):
    return {
        "modelo": modelo, "colunas_modelo": list(colunas),
        "colunas_constantes": list(constantes), "arquivos_treinamento": list(arquivos),
        "tempo_treinamento_segundos": tempo, "versao_sklearn": sklearn.__version__,
        **extras,
    }


def treinar_rf(X, y, parametros):
    validar_entradas(X)
    modelo = RandomForestClassifier(**parametros)
    inicio = perf_counter()
    modelo.fit(X, y)
    return modelo, perf_counter() - inicio


def treinar_if(X, parametros):
    modelo = IsolationForest(**parametros)
    inicio = perf_counter()
    modelo.fit(X)
    return modelo, perf_counter() - inicio


def transformar_entradas(pacote, X):
    validar_entradas(X)
    if list(X.columns) != list(pacote["colunas_modelo"]):
        raise ValueError("Use as entradas na mesma ordem do treinamento.")
    entrada = pacote["scaler"].transform(X) if "scaler" in pacote else X
    if "pca" in pacote:
        entrada = pacote["pca"].transform(entrada)
    return entrada


def pontuar(pacote, X):
    modelo = pacote["modelo"]
    entrada = transformar_entradas(pacote, X)
    if "scaler" in pacote:
        return -modelo.score_samples(entrada)
    indice = list(modelo.classes_).index(1)
    return modelo.predict_proba(entrada)[:, indice]


def prever_nativo(pacote, X):
    resultado = pacote["modelo"].predict(transformar_entradas(pacote, X))
    return (resultado == -1).astype(int) if "scaler" in pacote else np.asarray(resultado)
