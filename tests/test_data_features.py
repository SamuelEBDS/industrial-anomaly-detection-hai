import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from hai_tcc.data import ler_particao
from hai_tcc.features import ajustar_pca_normais, ajustar_scaler_normais, colunas_industriais, selecionar_colunas_treino, separar_entradas, validar_entradas


class DadosEntradasTests(unittest.TestCase):
    def test_leitura_preserva_origem_e_ordem_sem_excluir_linhas(self):
        with tempfile.TemporaryDirectory() as d:
            for name, start in [("b.csv.gz", 2), ("a.csv.gz", 0)]:
                pd.DataFrame({
                    "time": pd.date_range("2020-01-01", periods=2, freq="s") + pd.Timedelta(seconds=start),
                    "sensor": [10, 11], "attack": [0, 1],
                    "attack_P1": [0, 1], "attack_P2": [0, 0], "attack_P3": [0, 0],
                }).to_csv(Path(d) / name, index=False)
            df = ler_particao(d, ["b.csv.gz", "a.csv.gz"])
            X, y, meta = separar_entradas(df)
            self.assertEqual(len(X), 4)
            self.assertEqual(X.columns.tolist(), ["sensor"])
            self.assertEqual(y.tolist(), [0, 1, 0, 1])
            self.assertEqual(meta["arquivo_origem"].tolist(), ["a.csv.gz"] * 2 + ["b.csv.gz"] * 2)
            self.assertEqual(meta["indice_origem"].tolist(), [0, 1, 0, 1])

    def test_rotulos_e_identificadores_nunca_entram_em_X(self):
        self.assertEqual(colunas_industriais([
            "time", "sensor", "attack", "attack_P1", "attack_P2",
            "attack_P3", "attack_extra", "arquivo_origem", "indice_origem",
        ]), ["sensor"])
        with self.assertRaises(ValueError):
            validar_entradas(pd.DataFrame({"sensor": [1], "attack": [0]}))

    def test_constantes_sao_definidas_pelo_treino(self):
        treino = pd.DataFrame({"z": [1, 2, 3], "constante": [5, 5, 5], "a": [2, 4, 6]})
        _, colunas, constantes = selecionar_colunas_treino(treino)
        validacao = pd.DataFrame({"a": [0], "constante": [999], "z": [1]})
        self.assertEqual(colunas, ["z", "a"])
        self.assertEqual(constantes, ["constante"])
        self.assertEqual(validacao[colunas].columns.tolist(), ["z", "a"])

    def test_scaler_e_pca_aprendem_somente_normais_do_treino(self):
        X = pd.DataFrame({"a": [0., 2., 4., 1000.], "b": [0., 4., 8., 5000.]})
        y = pd.Series([0, 0, 0, 1])
        normais, scaler, escalado = ajustar_scaler_normais(X, y)
        pca, _ = ajustar_pca_normais(escalado, {"n_components": .95, "svd_solver": "full"})
        media = scaler.mean_.copy()
        componentes = pca.components_.copy()
        extremo = pd.DataFrame({"a": [100000.], "b": [-100000.]})
        pca.transform(scaler.transform(extremo))
        np.testing.assert_array_equal(media, [2., 4.])
        np.testing.assert_array_equal(scaler.mean_, media)
        np.testing.assert_array_equal(pca.components_, componentes)
        self.assertEqual(scaler.n_samples_seen_, 3)
        self.assertEqual(pca.n_samples_, 3)
        self.assertEqual(len(normais), 3)

    def test_rejeita_valores_nao_numericos_ausentes_e_infinitos(self):
        for valores in [["texto"], [np.nan], [np.inf]]:
            with self.subTest(valores=valores), self.assertRaises(ValueError):
                validar_entradas(pd.DataFrame({"sensor": valores}))

    def test_rejeita_horarios_repetidos(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"time": ["2020-01-01"] * 2, "sensor": [1, 2], "attack": [0, 1]}).to_csv(
                Path(d) / "a.csv.gz", index=False)
            with self.assertRaises(ValueError):
                ler_particao(d, ["a.csv.gz"])
