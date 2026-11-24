"""Subamostragem para a próxima etapa: somente normais de treino."""
import numpy as np


def selecionar_normais_treino(y, metadados, normais_por_ataque, semente):
    """Preserva ataques e distribui a cota de normais por arquivo e hora."""
    if not y.index.equals(metadados.index) or not y.index.is_unique:
        raise ValueError("Rótulos e metadados precisam de índices alinhados e únicos.")
    if not y.isin([0, 1]).all() or normais_por_ataque <= 0:
        raise ValueError("Rótulos ou proporção inválidos.")
    positivos = y.index[y.eq(1)].to_numpy()
    if len(positivos) == 0:
        raise ValueError("A subamostragem supervisionada precisa de ataques no treino.")
    grupos = metadados.loc[y.eq(0), ["arquivo_origem", "time"]].copy()
    grupos["hora"] = grupos["time"].dt.floor("h")
    total = min(len(grupos), int(len(positivos) * normais_por_ataque))
    if total == len(grupos):
        return y.index.copy()
    estratos = list(grupos.groupby(["arquivo_origem", "hora"], sort=True))
    tamanhos = np.array([len(g) for _, g in estratos])
    cotas_reais = total * tamanhos / tamanhos.sum()
    cotas = np.floor(cotas_reais).astype(int)
    falta = total - int(cotas.sum())
    maiores_restos = np.argsort(-(cotas_reais - cotas), kind="stable")
    cotas[maiores_restos[:falta]] += 1
    gerador = np.random.default_rng(semente)
    escolhidos = list(positivos)
    for (_, grupo), cota in zip(estratos, cotas):
        escolhidos.extend(gerador.choice(grupo.index.to_numpy(), size=int(cota), replace=False))
    # Retornar na ordem temporal original, sem modificar y nem os metadados.
    return y.index[y.index.isin(escolhidos)]
