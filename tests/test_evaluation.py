import unittest
import numpy as np
import pandas as pd
from hai_tcc import SUFIXOS
from hai_tcc.evaluation import aplicar_limiar, avaliar_intervalos, calcular_metricas, falsos_positivos_por_hora, identificar_intervalos, selecionar_limiar


class AvaliacaoTests(unittest.TestCase):
    def test_quatro_resultados_conhecidos(self):
        m = calcular_metricas([0, 0, 1, 1], [0, 1, 0, 1])
        for c in ["Verdadeiros positivos", "Verdadeiros negativos", "Falsos positivos", "Falsos negativos"]:
            self.assertEqual(m[c], 1)
        for c in ["Acurácia", "Precisão", "Recall", "F1-score", "Taxa de falsos positivos", "Taxa de falsos negativos"]:
            self.assertEqual(m[c], .5)

    def test_classe_ausente_nao_divide_por_zero(self):
        m = calcular_metricas([0, 0], [0, 0])
        self.assertEqual(m["Acurácia"], 1)
        self.assertEqual(m["Recall"], 0)
        self.assertTrue(np.isfinite(list(m.values())).all())

    def test_maior_limiar_em_empate_exato_no_F1(self):
        self.assertEqual(selecionar_limiar([1, 0, 0, 1], [.1, .2, .3, .4]), .4)

    def test_limiar_inclui_a_igualdade(self):
        np.testing.assert_array_equal(aplicar_limiar([.39, .4, .41], .4), [0, 1, 1])

    def test_calibracao_rejeita_uma_classe_e_scores_invalidos(self):
        for y, s in [([0, 0], [.1, .2]), ([0, 1], [.1, np.nan]), ([0, 1], [.1])]:
            with self.subTest(y=y, s=s), self.assertRaises(ValueError):
                selecionar_limiar(y, s)

    def test_intervalos_respeitam_lacuna_e_mudanca_de_arquivo(self):
        df = pd.DataFrame({
            "time": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:01",
                                    "2020-01-01 00:00:03", "2020-01-01 00:00:04"]),
            "arquivo_origem": ["a", "a", "a", "b"], "attack_real": [1, 1, 1, 1],
        })
        _, resumo = identificar_intervalos(df)
        self.assertEqual(resumo["registros_ataque"].tolist(), [2, 1, 1])

    def test_cobertura_atraso_e_falso_alarme_fora_do_trecho(self):
        df = pd.DataFrame({
            "time": pd.date_range("2020-01-01", periods=5, freq="s"),
            "arquivo_origem": ["a"] * 5, "attack_real": [0, 1, 1, 1, 0],
        })
        for nome, sufixo in SUFIXOS.items():
            df[f"previsao_{sufixo}"] = [1, 0, 0, 0, 1] if nome == "IF com PCA" else [1, 0, 1, 0, 1]
        analise, _ = identificar_intervalos(df)
        _, cobertura, atraso = avaliar_intervalos(analise)
        self.assertAlmostEqual(cobertura.loc[1, "RF"], 100 / 3)
        self.assertEqual(atraso.loc[1, "RF"], 1)
        self.assertEqual(cobertura.loc[1, "IF com PCA"], 0)
        self.assertTrue(np.isnan(atraso.loc[1, "IF com PCA"]))

    def test_taxa_por_hora_tem_denominador_apenas_normal(self):
        df = pd.DataFrame({
            "time": pd.date_range("2020-01-01", periods=3, freq="s"),
            "attack_real": [0, 1, 0],
            **{f"previsao_{s}": [1, 1, 1] for s in SUFIXOS.values()},
        })
        tabela = falsos_positivos_por_hora(df)
        self.assertEqual(tabela["Registros normais"].iloc[0], 2)
        self.assertEqual(tabela["FP RF"].iloc[0], 2)
        self.assertEqual(tabela["Taxa FP RF (%)"].iloc[0], 100)
