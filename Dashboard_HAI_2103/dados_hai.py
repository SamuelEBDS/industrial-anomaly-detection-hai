"""Leitura e comparação das medições e previsões históricas do HAI 21.03."""

import io
from pathlib import Path

import pandas as pd


PASTA_DADOS = Path(__file__).resolve().parent / "dados"
TOTAL = 92_401
ATAQUES = 2_177
MODELOS = {
    "Logística": "previsao_regressao_logistica",
    "IF com PCA": "previsao_if_com_pca",
    "Ensemble OU": "previsao_ensemble_ou",
}
SINAIS = (
    "P3_LIT01", "P3_PIT01", "P1_LIT01", "P1_PIT01", "P1_TIT01", "P1_FT01",
    "P2_SIT01", "P4_ST_PT01", "P4_ST_TT01",
)
CHAVE = ["arquivo_origem", "indice_origem", "time", "attack_real", "intervalo_id"]
COLUNAS_PREVISOES = CHAVE + list(MODELOS.values())


class DadosInvalidos(ValueError):
    """A entrada não pode ser alinhada com o teste histórico."""


def conferir_binaria(coluna, nome):
    if not coluna.isin([0, 1]).all():
        raise DadosInvalidos(f"A coluna {nome} deve conter somente 0 e 1.")


def ler_sinais():
    dados = pd.read_csv(PASTA_DADOS / "sinais_test5.csv.gz", parse_dates=["time"])
    if (
        len(dados) != TOTAL
        or dados["time"].isna().any()
        or not dados["time"].diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all()
    ):
        raise DadosInvalidos("Os registros de sinais não correspondem à sequência do test5.")
    conferir_binaria(dados["attack_real"], "attack_real")
    if dados["attack_real"].sum() != ATAQUES:
        raise DadosInvalidos("A contagem de rótulos de ataque não corresponde ao test5.")
    dados.insert(0, "indice_origem", dados.index)
    ataque = dados["attack_real"].eq(1)
    inicio = ataque & ~ataque.shift(fill_value=False)
    dados["intervalo_id"] = inicio.cumsum().where(ataque, 0)
    if dados["intervalo_id"].max() != 12:
        raise DadosInvalidos("A sequência de rótulos do test5 deve formar 12 episódios.")
    return dados


def ler_previsoes(origem, sinais, *, compactado=True):
    try:
        prev = pd.read_csv(
            origem,
            compression="gzip" if compactado else None,
            usecols=COLUNAS_PREVISOES,
            nrows=TOTAL + 1,
            parse_dates=["time"],
        )
    except (ValueError, OSError, EOFError) as exc:
        raise DadosInvalidos(
            "Escolha o arquivo teste_historico_ensemble_previsoes.csv.gz, "
            "gerado no fim do Notebook 06."
        ) from exc
    if len(prev) != TOTAL or not prev["arquivo_origem"].eq("test5.csv.gz").all():
        raise DadosInvalidos("O arquivo deve conter as 92.401 previsões do test5.")
    prev = prev.sort_values("indice_origem").reset_index(drop=True)
    for coluna in ("indice_origem", "time", "attack_real", "intervalo_id"):
        if not sinais[coluna].eq(prev[coluna]).all():
            raise DadosInvalidos("As linhas, horários, rótulos ou episódios não correspondem ao test5.")
    for coluna in MODELOS.values():
        conferir_binaria(prev[coluna], coluna)
    lr, ifp, conjunto = MODELOS.values()
    if not prev[conjunto].eq(prev[lr].eq(1) | prev[ifp].eq(1)).all():
        raise DadosInvalidos("A coluna do ensemble não corresponde à regra OU da Logística com o IF.")
    return sinais.join(prev[list(MODELOS.values())].astype("int8"))


def ler_referencia(sinais):
    return ler_previsoes(PASTA_DADOS / "previsoes_referencia_test5.csv.gz", sinais)


def ler_arquivo_enviado(conteudo, nome, sinais):
    compactado = nome.lower().endswith(".gz")
    if compactado and not conteudo.startswith(b"\x1f\x8b"):
        raise DadosInvalidos("O arquivo não é um CSV compactado em gzip.")
    return ler_previsoes(io.BytesIO(conteudo), sinais, compactado=compactado)


def episodios(dados):
    trechos = dados.loc[dados["intervalo_id"].gt(0)]
    return trechos.groupby("intervalo_id").agg(
        inicio=("indice_origem", "min"),
        fim=("indice_origem", "max"),
        horario=("time", "min"),
        registros=("attack_real", "size"),
    )


def resumir(dados):
    """Métricas por registro e tempo até o primeiro alerta por episódio detectado."""
    ataque = dados["attack_real"].eq(1)
    inicios = episodios(dados)["inicio"]
    resumo = {}
    for modelo, coluna in MODELOS.items():
        alerta = dados[coluna].eq(1)
        vp = int((ataque & alerta).sum())
        fp = int((~ataque & alerta).sum())
        fn = int((ataque & ~alerta).sum())
        primeiros = dados.loc[ataque & alerta].groupby("intervalo_id")["indice_origem"].min()
        atrasos = primeiros - inicios.loc[primeiros.index]
        resumo[modelo] = {
            "vp": vp,
            "fp": fp,
            "fn": fn,
            "f1": 2 * vp / (2 * vp + fp + fn) if vp else 0.0,
            "episodios": len(primeiros),
            "atraso": float(atrasos.median()) if len(atrasos) else None,
        }
    return resumo
