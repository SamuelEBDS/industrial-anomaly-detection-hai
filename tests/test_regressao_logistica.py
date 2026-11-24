import unittest

import numpy as np
import pandas as pd

from hai_tcc.classificacao.regressao_logistica import contar_rotulos_processos, treinar_logistica
from hai_tcc.models import empacotar, pontuar


class RegressaoLogisticaTests(unittest.TestCase):
    def test_escala_aprende_no_treino_e_nao_muda_ao_pontuar(self):
        X = pd.DataFrame({"sensor_a": [1., 2., 5., 8.], "sensor_b": [6., 7., 9., 8.]})
        y = pd.Series([0, 0, 1, 1])
        modelo, tempo = treinar_logistica(X, y)
        scaler = modelo.named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, [4., 7.5])
        self.assertEqual(scaler.n_samples_seen_, 4)
        pacote = empacotar(modelo, X.columns, [], ["treino"], tempo)
        outros = pd.DataFrame({"sensor_a": [-10., 20.], "sensor_b": [0., 100.]})
        pontuacao = pontuar(pacote, outros)
        self.assertGreater(pontuacao[1], pontuacao[0])
        np.testing.assert_allclose(scaler.mean_, [4., 7.5])

    def test_rotulos_usam_indice_de_origem_e_contam_sobreposicao(self):
        tempo = pd.date_range("2020-07-31 09:00", periods=3, freq="s")
        brutos = pd.DataFrame({"time": tempo, "attack": [0, 1, 1],
                               "attack_P1": [0, 0, 0], "attack_P2": [0, 1, 1], "attack_P3": [0, 1, 0]})
        base = brutos[["time", "attack"]].rename(columns={"attack": "attack_real"}).assign(
            indice_origem=[0, 1, 2], intervalo_id=[0, 6, 6],
        ).iloc[::-1].reset_index(drop=True)
        linha = contar_rotulos_processos(base, brutos).iloc[0]
        self.assertEqual(linha["intervalo_id"], 6)
        self.assertEqual(linha["registros_ataque"], 2)
        self.assertEqual((linha["attack_P1"], linha["attack_P2"], linha["attack_P3"]), (0, 2, 1))
        with self.assertRaises(AssertionError):
            contar_rotulos_processos(base.assign(attack_real=0), brutos)
