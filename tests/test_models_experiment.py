import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import yaml
from hai_tcc.experiment import abrir_experimento
from hai_tcc.models import pontuar


class RFClasseInvertida:
    classes_ = np.array([1, 0])
    def predict_proba(self, X):
        return np.tile([.8, .2], (len(X), 1))


class Isolamento:
    def score_samples(self, X):
        return -np.ones(len(X)) * .7


class ModeloExperimentoTests(unittest.TestCase):
    def test_RF_busca_a_coluna_da_classe_1(self):
        p = {"modelo": RFClasseInvertida(), "colunas_modelo": ["a"]}
        np.testing.assert_array_equal(pontuar(p, pd.DataFrame({"a": [0, 1]})), [.8, .8])

    def test_IF_pontua_sem_reajustar_scaler_e_rejeita_ordem_errada(self):
        treino = pd.DataFrame({"a": [0., 2.], "b": [1., 5.]})
        scaler = StandardScaler().fit(treino)
        pacote = {"modelo": Isolamento(), "colunas_modelo": ["a", "b"], "scaler": scaler}
        media = scaler.mean_.copy()
        np.testing.assert_array_equal(pontuar(pacote, treino * 1000), [.7, .7])
        np.testing.assert_array_equal(media, scaler.mean_)
        with self.assertRaises(ValueError):
            pontuar(pacote, treino[["b", "a"]])

    def montar(self, pasta):
        raiz = Path(pasta)
        (raiz / "src/hai_tcc").mkdir(parents=True)
        (raiz / "configs").mkdir()
        (raiz / "data").mkdir()
        (raiz / "data/origem.json").write_text(json.dumps({"arquivos": {}}))
        config = {"experimento": "teste", "versao_sklearn": sklearn.__version__, "dados": "data",
                  "particoes": {"treino": ["a"], "validacao": ["b"], "teste": ["c"]}}
        (raiz / "configs/baseline.yaml").write_text(yaml.safe_dump(config))
        return raiz, config

    def test_nao_aceita_sobreposicao_de_particoes(self):
        with tempfile.TemporaryDirectory() as d:
            raiz, config = self.montar(d)
            config["particoes"]["teste"] = ["a"]
            (raiz / "configs/baseline.yaml").write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                abrir_experimento(raiz)

    def test_manifesto_exige_ordem_e_invalida_saidas_apos_retreino(self):
        with tempfile.TemporaryDirectory() as d:
            raiz, _ = self.montar(d)
            e = abrir_experimento(raiz)
            with self.assertRaises(RuntimeError):
                e.registrar("validacao")
            for etapa in ["exploracao", "treinamento", "validacao", "teste", "analise"]:
                e.registrar(etapa)
            e.registrar("treinamento")
            e.exigir("treinamento")
            with self.assertRaises(RuntimeError):
                e.exigir("teste")

    def test_configuracao_alterada_nao_sobrescreve_experimento(self):
        with tempfile.TemporaryDirectory() as d:
            raiz, config = self.montar(d)
            abrir_experimento(raiz)
            config["novo_parametro"] = 123
            (raiz / "configs/baseline.yaml").write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                abrir_experimento(raiz)
