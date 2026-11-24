import unittest
import pandas as pd
from hai_tcc.sampling import selecionar_normais_treino


class AmostragemTests(unittest.TestCase):
    def setUp(self):
        self.y = pd.Series([0] * 24 + [1] * 4)
        self.meta = pd.DataFrame({
            "arquivo_origem": ["a"] * 12 + ["b"] * 16,
            "time": list(pd.date_range("2020-01-01", periods=6, freq="s"))
                    + list(pd.date_range("2020-01-01 01:00:00", periods=6, freq="s"))
                    + list(pd.date_range("2020-01-01 02:00:00", periods=16, freq="s")),
        })

    def test_preserva_todos_ataques_cota_e_estratos(self):
        indices = selecionar_normais_treino(self.y, self.meta, 2, 42)
        self.assertEqual(self.y.loc[indices].value_counts().to_dict(), {0: 8, 1: 4})
        self.assertEqual(set(self.y.index[self.y.eq(1)]) - set(indices), set())
        selecionados = self.meta.loc[indices[self.y.loc[indices].eq(0)]]
        contagens = selecionados.groupby(["arquivo_origem", selecionados["time"].dt.floor("h")]).size()
        self.assertEqual(contagens.tolist(), [2, 2, 4])
        self.assertTrue(indices.is_monotonic_increasing)

    def test_semente_fixa_e_repetivel(self):
        a = selecionar_normais_treino(self.y, self.meta, 2, 42)
        b = selecionar_normais_treino(self.y, self.meta, 2, 42)
        c = selecionar_normais_treino(self.y, self.meta, 2, 13)
        self.assertTrue(a.equals(b))
        self.assertFalse(a.equals(c))

    def test_nao_modifica_dados_originais(self):
        y, meta = self.y.copy(), self.meta.copy(deep=True)
        selecionar_normais_treino(self.y, self.meta, 1, 42)
        pd.testing.assert_series_equal(self.y, y)
        pd.testing.assert_frame_equal(self.meta, meta)

    def test_cota_alta_mantem_o_treino_inteiro(self):
        self.assertTrue(selecionar_normais_treino(self.y, self.meta, 20, 42).equals(self.y.index))

    def test_rejeita_desalinhamento_e_ausencia_de_ataques(self):
        with self.assertRaises(ValueError):
            selecionar_normais_treino(self.y, self.meta.iloc[::-1], 1, 42)
        with self.assertRaises(ValueError):
            selecionar_normais_treino(self.y * 0, self.meta, 1, 42)
