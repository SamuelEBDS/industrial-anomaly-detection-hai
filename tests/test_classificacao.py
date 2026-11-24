import unittest

import pandas as pd

from hai_tcc.classificacao.avaliacao import avaliar, ordenar_rf_validacao, preparar_avaliacao, resumir


class ClassificacaoTests(unittest.TestCase):
    def test_selecao_usa_media_das_sementes_e_rejeita_teste(self):
        linhas = pd.DataFrame({
            "particao": ["validacao"] * 4, "variante": ["a1", "a2", "a3", "b"],
            "configuracao": ["RF 1:20"] * 3 + ["RF ponderado"],
            "f1": [.9, .3, .3, .6], "episodios_totais": 1,
        })
        for col in ("precisao", "recall", "pr_auc", "average_precision", "falsos_positivos", "taxa_fp",
                    "episodios_detectados", "atraso_mediano_segundos"):
            linhas[col] = 0
        resumo = resumir(linhas)
        ranking = ordenar_rf_validacao(resumo)
        self.assertEqual(ranking.iloc[0]["configuracao"], "RF ponderado")
        self.assertAlmostEqual(ranking.iloc[1]["f1_media"], .5)
        with self.assertRaisesRegex(ValueError, "validação"):
            ordenar_rf_validacao(resumo.assign(particao="teste_historico"))

    def test_alerta_fora_do_episodio_e_atraso_do_primeiro_alerta(self):
        meta = pd.DataFrame({"time": pd.date_range("2020-07-31 09:00", periods=6, freq="s"),
                             "arquivo_origem": "teste", "indice_origem": range(6)})
        base, intervalos = preparar_avaliacao(meta, pd.Series([0, 1, 1, 0, 1, 0]))
        m, h, e, p = avaliar(base, intervalos, [.7, .4, .6, .8, .2, .1], .6)
        self.assertEqual(p.tolist(), [1, 0, 1, 1, 0, 0])
        self.assertEqual(m["falsos_positivos"], 2)
        self.assertEqual(m["episodios_detectados"], 1)
        self.assertEqual(h.iloc[0]["registros_normais"], 3)
        self.assertEqual(e["registros_detectados"].tolist(), [1, 0])
        self.assertEqual(e.iloc[0]["atraso_segundos"], 1)
        self.assertTrue(pd.isna(e.iloc[1]["atraso_segundos"]))
