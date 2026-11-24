"""Configuração, pacotes em disco e registro de cada execução."""
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import joblib
import sklearn
import yaml

from . import ARQUIVOS_MODELOS
from .data import hash_arquivo, ler_particao

ETAPAS = ("exploracao", "treinamento", "validacao", "teste", "analise")


def encontrar_projeto(inicio=None):
    inicio = Path(inicio or Path.cwd()).resolve()
    for pasta in (inicio, *inicio.parents):
        if (pasta / "configs" / "baseline.yaml").is_file() and (pasta / "src" / "hai_tcc").is_dir():
            return pasta
    raise FileNotFoundError("Abra o projeto extraído ou sua pasta notebooks.")


def _gravar_json(caminho, dados):
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporario.replace(caminho)


@dataclass
class Experimento:
    raiz: Path
    config: dict
    pasta: Path

    @property
    def dados(self):
        return self.raiz / self.config["dados"]

    @property
    def modelos(self):
        return self.pasta / "modelo_e_transformacoes"

    @property
    def resultados(self):
        return self.pasta / "previsoes_e_metricas"

    def exigir(self, etapa):
        atual = json.loads((self.pasta / "manifesto.json").read_text(encoding="utf-8"))
        if etapa not in atual["etapas"]:
            raise RuntimeError(f"Execute a etapa {etapa} neste experimento antes de continuar.")

    def registrar(self, etapa, **detalhes):
        if etapa not in ETAPAS:
            raise ValueError("Etapa desconhecida.")
        indice = ETAPAS.index(etapa)
        if indice:
            self.exigir(ETAPAS[indice - 1])
        caminho = self.pasta / "manifesto.json"
        atual = json.loads(caminho.read_text(encoding="utf-8"))
        # Uma nova execução invalida as etapas seguintes no mesmo experimento.
        for posterior in ETAPAS[indice:]:
            atual["etapas"].pop(posterior, None)
        atual["etapas"][etapa] = {
            "concluida_em": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            **detalhes,
        }
        _gravar_json(caminho, atual)

    def ler(self, particao, colunas=None):
        if particao not in ("treino", "validacao", "teste"):
            raise ValueError("Partição desconhecida.")
        return ler_particao(self.dados, self.config["particoes"][particao], colunas)

    def salvar_pacote(self, nome, pacote):
        joblib.dump(pacote, self.modelos / ARQUIVOS_MODELOS[nome], compress=3)

    def carregar_pacotes(self):
        self.exigir("treinamento")
        pacotes = {n: joblib.load(self.modelos / a) for n, a in ARQUIVOS_MODELOS.items()}
        colunas = [p["colunas_modelo"] for p in pacotes.values()]
        if not all(c == colunas[0] for c in colunas):
            raise ValueError("As entradas dos modelos são diferentes.")
        if any(p["versao_sklearn"] != sklearn.__version__ for p in pacotes.values()):
            raise ValueError("Versão do scikit-learn diferente da usada para treinar os modelos.")
        return pacotes


def abrir_experimento(raiz=None, configuracao=None):
    raiz = encontrar_projeto(raiz)
    configuracao = configuracao or os.environ.get("HAI_TCC_CONFIG", "configs/baseline.yaml")
    arquivo = (raiz / configuracao).resolve()
    config = yaml.safe_load(arquivo.read_text(encoding="utf-8"))
    if config.get("estado", "ativo") != "ativo":
        raise ValueError("Esta configuração é um plano para a próxima etapa; ainda não é executável.")
    particoes = config["particoes"]
    conjuntos = [set(particoes[p]) for p in ("treino", "validacao", "teste")]
    if any(not s for s in conjuntos) or any(conjuntos[i] & conjuntos[j] for i in range(3) for j in range(i)):
        raise ValueError("Treino, validação e teste precisam de arquivos distintos.")
    for nomes in particoes.values():
        if len(nomes) != len(set(nomes)):
            raise ValueError("Há arquivos repetidos na configuração.")
    if sklearn.__version__ != config["versao_sklearn"]:
        raise RuntimeError(f"Este experimento pede scikit-learn {config['versao_sklearn']}; instalado: {sklearn.__version__}.")
    identificador = config["experimento"]
    if not isinstance(identificador, str) or Path(identificador).name != identificador or identificador in (".", ".."):
        raise ValueError("Use um nome simples e distinto para cada experimento.")
    exp = Experimento(raiz, config, raiz / "runs" / identificador)
    exp.modelos.mkdir(parents=True, exist_ok=True)
    exp.resultados.mkdir(exist_ok=True)
    manifesto = exp.pasta / "manifesto.json"
    codigo = {p.name: hash_arquivo(p) for p in sorted((raiz / "src" / "hai_tcc").glob("*.py"))}
    identidade = {"config_sha256": hash_arquivo(arquivo), "codigo_sha256": codigo}
    if manifesto.exists():
        atual = json.loads(manifesto.read_text(encoding="utf-8"))
        if any(atual[k] != v for k, v in identidade.items()):
            raise ValueError("Código ou configuração mudou. Escolha um novo nome de experimento para preservar a execução anterior.")
    else:
        origem = json.loads((raiz / "data" / "origem.json").read_text(encoding="utf-8"))
        _gravar_json(manifesto, {
            **identidade, "configuracao": config, "python": platform.python_version(),
            "bibliotecas": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "PyYAML"]},
            "dados": origem, "etapas": {},
        })
    return exp
