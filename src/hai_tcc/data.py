"""Leitura dos CSVs, identidade das linhas e conferências de estrutura."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROTULOS = ("attack", "attack_P1", "attack_P2", "attack_P3")
METADADOS = ("time", "arquivo_origem", "indice_origem")


def hash_arquivo(caminho):
    with Path(caminho).open("rb") as arquivo:
        return hashlib.file_digest(arquivo, "sha256").hexdigest()


def garantir_dados(pasta, origem):
    """Baixa apenas os ausentes e verifica os hashes da revisão registrada."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/icsdataset/hai/{origem['revisao']}/hai-21.03"
    for nome, esperado in origem["arquivos"].items():
        destino = pasta / nome
        if not destino.is_file():
            temporario = destino.with_name(nome + ".part")
            print(f"Baixando {nome}...", flush=True)
            with urlopen(f"{url}/{nome}", timeout=90) as resposta, temporario.open("wb") as saida:
                while bloco := resposta.read(1024 * 1024):
                    saida.write(bloco)
            if hash_arquivo(temporario) != esperado["sha256"]:
                raise ValueError(f"Download com hash diferente: {nome}")
            temporario.replace(destino)
        if hash_arquivo(destino) != esperado["sha256"]:
            raise ValueError(f"Dados diferentes da revisão registrada: {nome}")
        print(f"Conferido: {nome}")


def resumir_arquivos(pasta, nomes):
    resumo = []
    for nome in nomes:
        dados = pd.read_csv(Path(pasta) / nome, usecols=["time", "attack"])
        tempo = pd.to_datetime(dados["time"], errors="raise")
        resumo.append({
            "arquivo": nome, "registros": len(dados),
            "normais": int(dados["attack"].eq(0).sum()),
            "ataques": int(dados["attack"].eq(1).sum()),
            "início": tempo.min(), "fim": tempo.max(),
        })
    return pd.DataFrame(resumo)


def verificar_qualidade(pasta, nomes, colunas):
    registros = []
    for nome in nomes:
        dados = pd.read_csv(Path(pasta) / nome)
        tempo = pd.to_datetime(dados["time"], errors="coerce")
        registros.append({
            "arquivo": nome, "mesmas_colunas": dados.columns.equals(pd.Index(colunas)),
            "valores_ausentes": int(dados.isna().sum().sum()),
            "horários_inválidos": int(tempo.isna().sum()),
            "horários_repetidos": int(tempo.duplicated().sum()),
            "ordem_temporal": tempo.is_monotonic_increasing,
            "rótulos_inválidos": int((~dados["attack"].isin([0, 1])).sum()),
        })
    return pd.DataFrame(registros)


def ler_particao(pasta, nomes, colunas=None):
    """Preserva arquivo e índice original; ordena no tempo sem excluir linhas."""
    if not nomes or len(set(nomes)) != len(nomes):
        raise ValueError("Informe arquivos distintos para a partição.")
    partes = []
    referencia = None
    for nome in nomes:
        if Path(nome).name != nome:
            raise ValueError("Use apenas o nome do CSV dentro da pasta de dados.")
        selecao = (lambda c: c not in ROTULOS[1:]) if colunas is None else ["time", "attack", *colunas]
        parte = pd.read_csv(Path(pasta) / nome, usecols=selecao, parse_dates=["time"])
        if referencia is not None and list(parte.columns) != referencia:
            raise ValueError(f"Colunas diferentes entre arquivos: {nome}")
        referencia = list(parte.columns)
        tempo = pd.to_datetime(parte["time"], errors="raise")
        if tempo.isna().any() or tempo.duplicated().any() or not tempo.is_monotonic_increasing:
            raise ValueError(f"Horários ausentes, repetidos ou fora de ordem: {nome}")
        if not parte["attack"].isin([0, 1]).all():
            raise ValueError(f"Rótulos diferentes de 0 e 1: {nome}")
        parte["time"] = tempo
        parte["arquivo_origem"] = nome
        parte["indice_origem"] = parte.index
        partes.append(parte)
    return pd.concat(partes, ignore_index=True).sort_values("time", kind="stable").reset_index(drop=True)
