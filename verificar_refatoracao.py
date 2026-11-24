"""Compara o experimento modular à reprodução da etapa 1."""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

import joblib
import numpy as np
import pandas as pd
from hai_tcc import ARQUIVOS_MODELOS, SUFIXOS


def verificar(nome="baseline_modular"):
    pasta = BASE / "runs" / nome
    atual = pasta / "previsoes_e_metricas"
    referencia = BASE / "referencia" / "reproducao_etapa_1"
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    if set(manifesto["etapas"]) != {"exploracao", "treinamento", "validacao", "teste", "analise"}:
        raise ValueError("Execute os cinco notebooks antes da conferência completa.")
    for original in sorted(referencia.glob("*.csv")):
        esperado = pd.read_csv(original)
        observado = pd.read_csv(atual / original.name)
        pd.testing.assert_frame_equal(observado, esperado, check_exact=False, rtol=1e-12, atol=1e-12)
    print("Todas as tabelas CSV da etapa 1 coincidem com a versão modular.")

    esperado = pd.read_csv(referencia / "teste_previsoes.csv.gz", float_precision="round_trip")
    observado = pd.read_csv(atual / "teste_previsoes.csv.gz", float_precision="round_trip")
    for coluna in ("time", "arquivo_origem", "attack_real"):
        np.testing.assert_array_equal(observado[coluna], esperado[coluna])
    for sufixo in SUFIXOS.values():
        np.testing.assert_array_equal(observado[f"previsao_{sufixo}"], esperado[f"previsao_{sufixo}"])
        for campo in ("pontuacao", "limiar"):
            np.testing.assert_allclose(observado[f"{campo}_{sufixo}"], esperado[f"{campo}_{sufixo}"],
                                       rtol=0, atol=1e-12)
    print(f"{len(observado):,} registros: previsões idênticas; pontuações e cortes coincidem até 1e-12.")

    esperado = json.loads((referencia / "limiares.json").read_text())
    limiares = joblib.load(pasta / "modelo_e_transformacoes/limiares_validacao.joblib")["limiares"]
    np.testing.assert_allclose(list(limiares.values()), list(esperado.values()), rtol=0, atol=1e-12)
    protocolo = json.loads((BASE / "referencia/protocolo.json").read_text())
    for modelo, arquivo in ARQUIVOS_MODELOS.items():
        pacote = joblib.load(pasta / "modelo_e_transformacoes" / arquivo)
        if pacote["colunas_modelo"] != protocolo["colunas_modelo"]:
            raise ValueError(f"{modelo}: ordem das 60 entradas diferente.")
        if pacote["colunas_constantes"] != protocolo["colunas_constantes"]:
            raise ValueError(f"{modelo}: constantes diferentes.")
    tabela = pd.read_csv(atual / "teste_metricas.csv", index_col=0)
    print("F1 no teste:", tabela.loc["F1-score"].round(4).to_dict())
    print("REFATORAÇÃO CONFERIDA.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", default="baseline_modular")
    verificar(parser.parse_args().experimento)
