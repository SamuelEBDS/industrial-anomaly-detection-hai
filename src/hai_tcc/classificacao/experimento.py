"""Executa o plano de balanceamento e guarda cada rodada em uma pasta nova."""
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
import json
import platform
import re
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from .. import SUFIXOS
from ..data import hash_arquivo, ler_particao
from ..evaluation import aplicar_limiar, selecionar_limiar
from ..experiment import ETAPAS
from ..features import selecionar_colunas_treino, separar_entradas
from ..models import empacotar, pontuar, treinar_rf
from ..sampling import selecionar_normais_treino
from .avaliacao import avaliar, investigar_fp, ordenar_rf_validacao, preparar_avaliacao, resumir
from .relatorio import gerar_relatorio


@dataclass(frozen=True)
class Variante:
    id: str
    configuracao: str
    normais_por_ataque: int | None = None
    semente: int | None = None


REFERENCIAS = {
    "RF": Variante("rf_ponderado", "RF ponderado"),
    "IF sem PCA": Variante("if_sem_pca", "IF sem PCA"),
    "IF com PCA": Variante("if_com_pca", "IF com PCA"),
}


def variantes_rf(plano):
    sementes, proporcoes = plano["sementes"], plano["proporcoes_ataque_normal"]
    if (len(sementes) != 3 or len(set(sementes)) != 3
        or any(type(s) is not int or s < 0 for s in sementes)
        or len(proporcoes) != len(set(proporcoes)) or not proporcoes
        or any(type(p) is not int or p <= 0 for p in proporcoes)):
        raise ValueError("Defina três sementes distintas e proporções positivas e distintas.")
    if (plano["selecionar_configuracao_em"] != "validacao" or plano["test5"] != "comparacao_historica"
        or plano["preservar_todos_ataques"] is not True or plano["estratos"] != ["arquivo_origem", "hora"]
        or plano["pesos_de_classe_com_subamostragem"] is not None):
        raise ValueError("O plano precisa preservar o protocolo de avaliação do balanceamento.")
    return [Variante("rf_sem_pesos", "RF sem pesos")] + [
        Variante(f"rf_1_{p}_s{s}", f"RF 1:{p}", p, s) for p in proporcoes for s in sementes
    ]


def _agora():
    return datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()


def _salvar_json(caminho, dados):
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(json.dumps(dados, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporario.replace(caminho)


def _ler_json(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def _ler_particao_conferida(pasta_dados, nomes, origem, colunas=None):
    for nome in nomes:
        caminho = pasta_dados / nome
        if not caminho.is_file():
            raise FileNotFoundError(f"{nome} não encontrado. Execute a exploração do projeto modular para preparar os dados.")
        if hash_arquivo(caminho) != origem["arquivos"][nome]["sha256"]:
            raise ValueError(f"Hash de {nome} difere da revisão dos dados usada na referência.")
    return ler_particao(pasta_dados, nomes, colunas)


def _ler_referencia(caminho, base, limiares):
    referencias = pd.read_csv(caminho, parse_dates=["time"], float_precision="round_trip")
    chaves = ["time", "arquivo_origem", "indice_origem", "attack_real"]
    pd.testing.assert_frame_equal(base[chaves], referencias[chaves], check_dtype=False)
    for nome, sufixo in SUFIXOS.items():
        limiar = limiares[nome]
        np.testing.assert_allclose(referencias[f"limiar_{sufixo}"], limiar, atol=1e-12, rtol=0)
        scores = referencias[f"pontuacao_{sufixo}"].to_numpy()
        np.testing.assert_array_equal(aplicar_limiar(scores, limiar), referencias[f"previsao_{sufixo}"])
        yield REFERENCIAS[nome], scores, limiar


def _novas_avaliacoes(nome, destino, meta, y, registros_treino, positivos_treino):
    base, intervalos = preparar_avaliacao(meta, y)
    previsoes = base.copy()
    metricas, horas, episodios = [], [], []

    def registrar(variante, scores, limiar):
        metrica, por_hora, por_episodio, previsto = avaliar(base, intervalos, scores, limiar)
        ids = {"particao": nome, "variante": variante.id, "configuracao": variante.configuracao,
               "semente_amostragem": variante.semente}
        positivos = 0 if variante.id.startswith("if_") else positivos_treino
        normais = (variante.normais_por_ataque * positivos if variante.normais_por_ataque
                   else registros_treino - positivos_treino)
        metricas.append({**ids, "normais_treino": normais, "ataques_treino": positivos, **metrica})
        horas.append(por_hora.assign(**ids))
        episodios.append(por_episodio.assign(**ids))
        previsoes[f"pontuacao_{variante.id}"] = scores
        previsoes[f"previsao_{variante.id}"] = previsto

    def salvar():
        m = pd.DataFrame(metricas).astype({"semente_amostragem": "Int64"})
        h = pd.concat(horas, ignore_index=True).astype({"semente_amostragem": "Int64"})
        e = pd.concat(episodios, ignore_index=True).astype({"semente_amostragem": "Int64"})
        for tabela, sufixo in [(m, "metricas"), (h, "falsos_positivos_por_hora"), (e, "episodios")]:
            tabela.to_csv(destino / f"{nome}_{sufixo}.csv", index=False)
        previsoes.to_csv(destino / f"{nome}_previsoes.csv.gz", index=False, compression="gzip")
        return m, h, e

    return base, registrar, salvar


def executar(raiz, nome, nome_referencia):
    if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", n) for n in (nome, nome_referencia)):
        raise ValueError("Use um nome simples para cada execução.")
    pasta_referencia = raiz / "runs" / nome_referencia
    caminho_referencia = pasta_referencia / "manifesto.json"
    if not caminho_referencia.is_file():
        raise FileNotFoundError("Execute e confira os cinco notebooks da versão modular primeiro.")
    referencia = _ler_json(caminho_referencia)
    if set(referencia["etapas"]) != set(ETAPAS):
        raise ValueError("A execução modular usada como referência precisa estar completa.")
    config = referencia["configuracao"]
    if config["versao_sklearn"] != sklearn.__version__:
        raise RuntimeError(f"Use scikit-learn {config['versao_sklearn']} como na referência.")
    protocolo = _ler_json(raiz / "referencia/protocolo.json")
    if (config["particoes"] != {"treino": protocolo["treino"], "validacao": [protocolo["validacao"]],
                               "teste": [protocolo["teste"]]}
        or config["rf"]["n_estimators"] != 100 or config["rf"]["class_weight"] != "balanced_subsample"):
        raise ValueError("A referência deve usar a divisão original e o RF ponderado de 100 árvores.")
    origem = _ler_json(raiz / "data/origem.json")
    if origem != referencia["dados"]:
        raise ValueError("A revisão dos dados difere da referência.")
    plano_caminho = raiz / "configs/classificacao.yaml"
    plano = yaml.safe_load(plano_caminho.read_text(encoding="utf-8"))
    variantes = variantes_rf(plano)
    parametros_rf = {**config["rf"], "class_weight": None}
    dados = raiz / config["dados"]
    particoes = config["particoes"]
    limiares_referencia = referencia["etapas"]["validacao"]["limiares"]
    origem_resultados = pasta_referencia / "previsoes_e_metricas"
    ref_validacao = origem_resultados / "validacao_previsoes.csv.gz"

    pasta = raiz / "runs" / nome
    pasta.mkdir(exist_ok=False)
    destino, modelos, amostras = (pasta / sub for sub in ("previsoes_e_metricas", "modelo_e_transformacoes", "amostras"))
    for sub in (destino, modelos, amostras):
        sub.mkdir()
    caminho_manifesto = pasta / "manifesto.json"
    manifesto = {
        "inicio": _agora(), "referencia": nome_referencia, "referencia_sha256": hash_arquivo(caminho_referencia),
        "dados": origem, "particoes": particoes, "python": platform.python_version(),
        "bibliotecas": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "PyYAML")},
        "plano_sha256": hash_arquivo(plano_caminho), "parametros_rf_sem_pesos": parametros_rf,
        "plano": plano, "variantes_novas": [asdict(v) for v in variantes],
        "codigo_sha256": {str(p.relative_to(raiz)): hash_arquivo(p)
                          for p in [raiz / "executar_balanceamento.py", *sorted((raiz / "src/hai_tcc").rglob("*.py"))]},
        "etapas": {},
    }
    _salvar_json(caminho_manifesto, manifesto)

    X, y, meta = separar_entradas(_ler_particao_conferida(dados, particoes["treino"], origem))
    X, colunas, constantes = selecionar_colunas_treino(X)
    if len(colunas) != 60 or colunas != protocolo["colunas_modelo"] or constantes != protocolo["colunas_constantes"]:
        raise ValueError("A ordem das 60 variáveis ou a lista de constantes mudou.")
    X_val, y_val, meta_val = separar_entradas(
        _ler_particao_conferida(dados, particoes["validacao"], origem, colunas), colunas,
    )
    num_ataques = int(y.eq(1).sum())
    if len(y) != 270003 or num_ataques != 5613 or len(y_val) != referencia["etapas"]["validacao"]["registros"]:
        raise ValueError("Contagens de treino ou validação diferentes da referência.")
    manifesto["colunas_modelo"] = colunas
    manifesto["colunas_constantes"] = constantes
    manifesto["contagens"] = {"treino": {"normais": len(y) - num_ataques, "ataques": num_ataques},
                               "validacao": {"normais": int(y_val.eq(0).sum()), "ataques": int(y_val.eq(1).sum())}}

    base_val, registrar_val, salvar_val = _novas_avaliacoes("validacao", destino, meta_val, y_val, len(y), num_ataques)
    for variante, scores, limiar in _ler_referencia(ref_validacao, base_val, limiares_referencia):
        registrar_val(variante, scores, limiar)
    for i, variante in enumerate(variantes, start=1):
        print(f"[{i}/{len(variantes)}] {variante.configuracao}, semente da amostra: {variante.semente}", flush=True)
        if variante.normais_por_ataque is None:
            indices = y.index
        else:
            indices = selecionar_normais_treino(y, meta, variante.normais_por_ataque, variante.semente)
        y_amostra = y.loc[indices]
        if int(y_amostra.eq(1).sum()) != num_ataques or not indices.is_unique:
            raise ValueError("A amostra deve preservar todos os registros de ataque sem duplicatas.")
        if variante.normais_por_ataque is not None:
            if int(y_amostra.eq(0).sum()) != variante.normais_por_ataque * num_ataques:
                raise ValueError("Quantidade de normais diferente da planejada.")
            meta.loc[indices].assign(attack=y_amostra).to_csv(
                amostras / f"{variante.id}.csv.gz", index=False, compression="gzip",
            )
        modelo, tempo = treinar_rf(X.loc[indices], y_amostra, parametros_rf)
        pacote = empacotar(modelo, colunas, constantes, particoes["treino"], tempo, variante=asdict(variante),
                            normais_treino=int(y_amostra.eq(0).sum()), ataques_treino=num_ataques)
        scores = pontuar(pacote, X_val)
        limiar = selecionar_limiar(y_val, scores)
        pacote["limiar_validacao"] = limiar
        joblib.dump(pacote, modelos / f"{variante.id}.joblib", compress=3)
        registrar_val(variante, scores, limiar)
    m_val, _, _ = salvar_val()
    classificacao = ordenar_rf_validacao(resumir(m_val))
    classificacao.insert(0, "posicao", np.arange(1, len(classificacao) + 1))
    classificacao.to_csv(destino / "ranking_rf_validacao.csv", index=False)
    melhor = classificacao.iloc[0]["configuracao"]
    melhor_amostrado = classificacao.loc[classificacao["configuracao"].str.startswith("RF 1:")].iloc[0]["configuracao"]
    selecao = {
        "particao": "validacao", "arquivo": protocolo["validacao"],
        "criterio": "Maior F1 médio; em empate, menos FP médios e depois ordem alfabética.",
        "melhor_rf": melhor, "melhor_rf_subamostrado": melhor_amostrado,
        "limiares": dict(zip(m_val["variante"], m_val["limiar"])), "fixada_em": _agora(),
    }
    _salvar_json(destino / "selecao_validacao.json", selecao)
    manifesto["etapas"]["validacao"] = {**selecao, "previsoes_referencia_sha256": hash_arquivo(ref_validacao)}
    _salvar_json(caminho_manifesto, manifesto)
    print(f"Escolha registrada em test4: {melhor} (melhor subamostrado: {melhor_amostrado}).", flush=True)

    # Nenhum dado de test5 é lido acima. O teste não realimenta a seleção.
    print("Avaliando test5 com os limiares já fixados...", flush=True)
    X_teste, y_teste, meta_teste = separar_entradas(
        _ler_particao_conferida(dados, particoes["teste"], origem, colunas), colunas,
    )
    if len(y_teste) != referencia["etapas"]["teste"]["registros"]:
        raise ValueError("Contagem do teste histórico diferente da referência.")
    base_teste, registrar_teste, salvar_teste = _novas_avaliacoes(
        "teste_historico", destino, meta_teste, y_teste, len(y), num_ataques,
    )
    ref_teste = origem_resultados / "teste_previsoes.csv.gz"
    for variante, scores, limiar in _ler_referencia(ref_teste, base_teste, limiares_referencia):
        registrar_teste(variante, scores, limiar)
    for variante in variantes:
        pacote = joblib.load(modelos / f"{variante.id}.joblib")
        if pacote["limiar_validacao"] != selecao["limiares"][variante.id]:
            raise ValueError("O limiar salvo não corresponde à seleção na validação.")
        registrar_teste(variante, pontuar(pacote, X_teste), pacote["limiar_validacao"])
    m_teste, horas_teste, episodios_teste = salvar_teste()
    resumo = resumir(pd.concat([m_val, m_teste], ignore_index=True))
    resumo.to_csv(destino / "comparacao_configuracoes.csv", index=False)
    top5, sinais = investigar_fp(horas_teste, base_teste, X_teste, base_val, X_val)
    top5.to_csv(destino / "cinco_horas_mais_fp_rf_ponderado.csv", index=False)
    sinais.to_csv(destino / "sinais_normais_cinco_horas.csv", index=False)
    intervalos_6_9 = episodios_teste.loc[episodios_teste["intervalo_id"].isin([6, 9])]
    intervalos_6_9.to_csv(destino / "intervalos_6_e_9.csv", index=False)
    gerar_relatorio(destino, selecao, resumo, m_val, m_teste, top5, sinais, intervalos_6_9)
    manifesto["contagens"]["teste_historico"] = {"normais": int(y_teste.eq(0).sum()), "ataques": int(y_teste.eq(1).sum())}
    manifesto["etapas"]["teste_historico"] = {"concluida_em": _agora(), "registros": len(y_teste),
                                              "previsoes_referencia_sha256": hash_arquivo(ref_teste)}
    _salvar_json(caminho_manifesto, manifesto)
    print(f"Resultados: {destino}\nBALANCEAMENTO CONCLUÍDO.", flush=True)
