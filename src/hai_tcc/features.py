"""Escolha das entradas e transformações aprendidas somente no treino."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import METADADOS, ROTULOS


def colunas_industriais(colunas):
    return [c for c in colunas if c not in (*METADADOS, *ROTULOS) and not c.lower().startswith("attack_")]


def validar_entradas(X):
    if not X.columns.is_unique or not X.index.is_unique:
        raise ValueError("As entradas precisam de nomes e índices distintos.")
    if list(X.columns) != colunas_industriais(X.columns):
        raise ValueError("Rótulos e metadados não podem entrar em X.")
    if len(X.select_dtypes(exclude="number").columns):
        raise ValueError("Há entradas não numéricas.")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Há entradas ausentes ou infinitas.")


def separar_entradas(dados, colunas=None):
    nomes = colunas_industriais(dados.columns) if colunas is None else list(colunas)
    X = dados.loc[:, nomes].copy()
    validar_entradas(X)
    y = dados["attack"].copy()
    metadados = dados.loc[:, list(METADADOS)].copy()
    return X, y, metadados


def selecionar_colunas_treino(X):
    validar_entradas(X)
    constantes = X.nunique().loc[lambda n: n <= 1].index.tolist()
    colunas = [c for c in X if c not in constantes]
    if not colunas:
        raise ValueError("O treino não tem nenhuma variável não constante.")
    return X.loc[:, colunas].copy(), colunas, constantes


def ajustar_scaler_normais(X_treino, y_treino):
    validar_entradas(X_treino)
    if not X_treino.index.equals(y_treino.index):
        raise ValueError("Entradas e rótulos do treino estão desalinhados.")
    normais = X_treino.loc[y_treino.eq(0)].copy()
    if normais.empty:
        raise ValueError("São necessários registros normais para ajustar o IF.")
    scaler = StandardScaler()
    return normais, scaler, scaler.fit_transform(normais)


def ajustar_pca_normais(normais_escalados, parametros):
    pca = PCA(**parametros)
    # fit_transform mantém a mesma operação numérica da referência.
    return pca, pca.fit_transform(normais_escalados)
