"""Avalia um segundo classificador supervisionado com a divisão temporal original."""
from importlib.metadata import version
import platform
import re
from time import perf_counter
import warnings

import joblib
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data import hash_arquivo
from ..evaluation import selecionar_limiar
from ..features import selecionar_colunas_treino, separar_entradas, validar_entradas
from ..models import empacotar, pontuar
from .avaliacao import resumir
from .experimento import Variante, _agora, _ler_json, _ler_particao_conferida, _novas_avaliacoes, _salvar_json
from .relatorio_logistica import gerar_sintese


VARIANTE = Variante("regressao_logistica", "Regressão Logística sem pesos")
PARAMETROS = {"solver": "lbfgs", "C": 1.0, "l1_ratio": 0.0, "class_weight": None, "max_iter": 1000, "tol": 1e-6}
ROTULOS_PROCESSOS = ["attack_P1", "attack_P2", "attack_P3"]
CHAVES = ["time", "arquivo_origem", "indice_origem", "attack_real"]


def treinar_logistica(X, y):
    validar_entradas(X)
    if not X.index.equals(y.index) or set(y.unique()) != {0, 1}:
        raise ValueError("O treino precisa de entradas alinhadas aos rótulos binários 0 e 1.")
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(**PARAMETROS))
    inicio = perf_counter()
    with threadpool_limits(limits=2, user_api="blas"), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, y)
    return pipeline, perf_counter() - inicio


def contar_rotulos_processos(base_teste, dados_brutos):
    """Cruza os rótulos de processo com episódios depois da avaliação."""
    indices = base_teste["indice_origem"]
    if len(dados_brutos) != len(indices) or sorted(indices) != list(range(len(indices))):
        raise ValueError("Não foi possível alinhar os rótulos de processo pelo índice de origem.")
    brutos = dados_brutos.iloc[indices.to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        base_teste[["time", "attack_real"]].reset_index(drop=True),
        brutos[["time", "attack"]].rename(columns={"attack": "attack_real"}), check_dtype=False,
    )
    if not brutos[ROTULOS_PROCESSOS].isin([0, 1]).all().all():
        raise ValueError("Há rótulos de processo inválidos.")
    dados = base_teste[["intervalo_id", "time", "attack_real"]].copy()
    dados[ROTULOS_PROCESSOS] = brutos[ROTULOS_PROCESSOS].to_numpy()
    return dados.loc[dados["attack_real"].eq(1)].groupby("intervalo_id", as_index=False).agg(
        inicio=("time", "min"), fim=("time", "max"), registros_ataque=("attack_real", "size"),
        **{rotulo: (rotulo, "sum") for rotulo in ROTULOS_PROCESSOS},
    )


def _comparar_registros(base, caminho):
    antigos = pd.read_csv(caminho, usecols=CHAVES, parse_dates=["time"])
    pd.testing.assert_frame_equal(base[CHAVES], antigos[CHAVES], check_dtype=False)


def _contagens(y, referencia, particao):
    totais = {"normais": int(y.eq(0).sum()), "ataques": int(y.eq(1).sum())}
    if totais != referencia["contagens"][particao]:
        raise ValueError(f"A contagem de {particao} difere do experimento de balanceamento.")
    return totais


def executar(raiz, nome, nome_balanceamento):
    if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", n) for n in (nome, nome_balanceamento)):
        raise ValueError("Use nomes simples para as pastas de execução.")
    anterior = raiz / "runs" / nome_balanceamento
    ref_manifesto = anterior / "manifesto.json"
    if not ref_manifesto.is_file():
        raise FileNotFoundError(f"Execução de balanceamento não encontrada em {anterior}.")
    referencia = _ler_json(ref_manifesto)
    if set(referencia.get("etapas", {})) != {"validacao", "teste_historico"} or "parametros_rf_sem_pesos" not in referencia:
        raise ValueError("Escolha uma execução concluída da etapa de balanceamento.")
    if referencia["bibliotecas"]["scikit-learn"] != version("scikit-learn"):
        raise ValueError("Use a mesma versão do scikit-learn usada no balanceamento.")
    protocolo = _ler_json(raiz / "referencia/protocolo.json")
    particoes = {"treino": protocolo["treino"], "validacao": [protocolo["validacao"]], "teste": [protocolo["teste"]]}
    if particoes != referencia["particoes"]:
        raise ValueError("A divisão temporal precisa ser a mesma do experimento de balanceamento.")
    origem = _ler_json(raiz / "data/origem.json")
    if origem != referencia["dados"]:
        raise ValueError("A revisão dos dados difere da referência do balanceamento.")
    config = yaml.safe_load((raiz / "configs/baseline.yaml").read_text(encoding="utf-8"))
    dados = raiz / config["dados"]
    ref_saida = anterior / "previsoes_e_metricas"
    ref_comparacao = ref_saida / "comparacao_configuracoes.csv"

    pasta = raiz / "runs" / nome
    pasta.mkdir(exist_ok=False)
    saida, modelos = pasta / "previsoes_e_metricas", pasta / "modelo_e_transformacoes"
    saida.mkdir()
    modelos.mkdir()
    caminho_manifesto = pasta / "manifesto.json"
    manifesto = {
        "inicio": _agora(), "referencia_balanceamento": nome_balanceamento,
        "referencia_manifesto_sha256": hash_arquivo(ref_manifesto), "dados": origem, "particoes": particoes,
        "python": platform.python_version(),
        "bibliotecas": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "PyYAML", "threadpoolctl")},
        "parametros": PARAMETROS, "padronizacao": "StandardScaler ajustado em todo o treino",
        "subamostragem": False, "etapas": {}, "contagens": {},
        "codigo_sha256": {str(p.relative_to(raiz)): hash_arquivo(p)
                          for p in [raiz / "executar_regressao_logistica.py", *sorted((raiz / "src/hai_tcc").rglob("*.py"))]},
    }
    _salvar_json(caminho_manifesto, manifesto)

    print("[1/3] Treinando com test1, test2 e test3...", flush=True)
    X, y, _ = separar_entradas(_ler_particao_conferida(dados, particoes["treino"], origem))
    X, colunas, constantes = selecionar_colunas_treino(X)
    if (len(colunas) != 60 or colunas != protocolo["colunas_modelo"] or colunas != referencia["colunas_modelo"]
        or constantes != protocolo["colunas_constantes"] or constantes != referencia["colunas_constantes"]):
        raise ValueError("A ordem das 60 entradas ou a lista de constantes difere da referência.")
    manifesto["contagens"]["treino"] = _contagens(y, referencia, "treino")
    modelo, tempo = treinar_logistica(X, y)
    iteracoes = int(modelo.named_steps["logisticregression"].n_iter_[0])
    pacote = empacotar(modelo, colunas, constantes, particoes["treino"], tempo, parametros=PARAMETROS)
    n_treino, positivos = len(y), int(y.eq(1).sum())
    del X, y
    manifesto["colunas_modelo"] = colunas
    manifesto["colunas_constantes"] = constantes
    manifesto["etapas"]["treino"] = {"concluida_em": _agora(), "iteracoes": iteracoes, "segundos": tempo}
    _salvar_json(caminho_manifesto, manifesto)

    print("[2/3] Escolhendo limiar pelo maior F1 no test4...", flush=True)
    X_val, y_val, meta_val = separar_entradas(_ler_particao_conferida(dados, particoes["validacao"], origem, colunas), colunas)
    manifesto["contagens"]["validacao"] = _contagens(y_val, referencia, "validacao")
    scores_val = pontuar(pacote, X_val)
    limiar = selecionar_limiar(y_val, scores_val)
    base_val, registrar_val, salvar_val = _novas_avaliacoes("validacao", saida, meta_val, y_val, n_treino, positivos)
    _comparar_registros(base_val, ref_saida / "validacao_previsoes.csv.gz")
    registrar_val(VARIANTE, scores_val, limiar)
    m_val, _, _ = salvar_val()
    pacote["limiar_validacao"] = limiar
    joblib.dump(pacote, modelos / "regressao_logistica.joblib", compress=3)
    selecao = {"particao": "validacao", "arquivo": protocolo["validacao"], "limiar": limiar,
               "criterio": "Maior F1; em empate, maior limiar.", "fixada_em": _agora()}
    _salvar_json(saida / "selecao_validacao.json", selecao)
    manifesto["etapas"]["validacao"] = selecao
    _salvar_json(caminho_manifesto, manifesto)
    print(f"Limiar fixado: {limiar:.10f}. F1 de validação: {m_val.iloc[0]['f1']:.4f}.", flush=True)

    # O limiar e o modelo já estão salvos antes de ler dados ou métricas de test5.
    print("[3/3] Avaliando test5 com o limiar fixado...", flush=True)
    X_teste, y_teste, meta_teste = separar_entradas(_ler_particao_conferida(dados, particoes["teste"], origem, colunas), colunas)
    manifesto["contagens"]["teste_historico"] = _contagens(y_teste, referencia, "teste_historico")
    base_teste, registrar_teste, salvar_teste = _novas_avaliacoes("teste_historico", saida, meta_teste, y_teste, n_treino, positivos)
    _comparar_registros(base_teste, ref_saida / "teste_historico_previsoes.csv.gz")
    registrar_teste(VARIANTE, pontuar(pacote, X_teste), limiar)
    m_teste, _, e_teste = salvar_teste()

    comparacao = pd.concat([pd.read_csv(ref_comparacao), resumir(pd.concat([m_val, m_teste]))], ignore_index=True)
    comparacao.to_csv(saida / "comparacao_modelos.csv", index=False)
    dados_rotulos = pd.read_csv(dados / protocolo["teste"], usecols=["time", "attack", *ROTULOS_PROCESSOS], parse_dates=["time"])
    rotulos = contar_rotulos_processos(base_teste, dados_rotulos)
    rotulos.to_csv(saida / "rotulos_processos_test5.csv", index=False)
    episodios_anteriores = pd.read_csv(ref_saida / "teste_historico_episodios.csv",
                                     dtype={"semente_amostragem": "Int64"},
                                     parse_dates=["inicio", "fim", "primeiro_alerta"])
    episodios = pd.concat([episodios_anteriores, e_teste], ignore_index=True)
    episodios.loc[episodios["intervalo_id"].isin([6, 9])].to_csv(saida / "intervalos_6_e_9_modelos.csv", index=False)
    gerar_sintese(saida, comparacao, m_teste.iloc[0], rotulos, e_teste, VARIANTE.configuracao,
                  referencia["etapas"]["validacao"]["melhor_rf_subamostrado"])
    manifesto["etapas"]["teste_historico"] = {"concluida_em": _agora(), "registros": len(y_teste), "limiar": limiar}
    manifesto["referencia_comparacao_sha256"] = hash_arquivo(ref_comparacao)
    _salvar_json(caminho_manifesto, manifesto)
    print(f"F1 no teste histórico: {m_teste.iloc[0]['f1']:.4f}.", flush=True)
    print(f"Resultados: {saida}\nREGRESSÃO LOGÍSTICA CONCLUÍDA.", flush=True)
