"""Executa testes, cinco notebooks e conferência em uma pasta nova."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import shutil
import subprocess
import sys
import unittest
from zoneinfo import ZoneInfo
import yaml

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))


def testes():
    suite = unittest.defaultTestLoader.discover(str(BASE / "tests"))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit("Os testes falharam. Nenhum modelo novo foi treinado.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testes", action="store_true", help="Roda somente os testes pequenos.")
    parser.add_argument("--nome", help="Nome da execução nova; a pasta ainda não pode existir.")
    args = parser.parse_args()
    testes()
    if args.testes:
        return
    nome = args.nome or datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("baseline_%Y%m%d_%H%M%S")
    if Path(nome).name != nome or nome in (".", ".."):
        raise SystemExit("Use um nome simples para a execução.")
    pasta = BASE / "runs" / nome
    pasta.mkdir(parents=True, exist_ok=False)
    with (BASE / "configs/baseline.yaml").open(encoding="utf-8") as arquivo:
        config = yaml.safe_load(arquivo)
    config["experimento"] = nome
    configuracao = pasta / "configuracao.yaml"
    configuracao.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    notebooks = pasta / "notebooks"
    shutil.copytree(BASE / "notebooks", notebooks)
    ambiente = os.environ.copy()
    ambiente["HAI_TCC_CONFIG"] = str(configuracao)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        ambiente[var] = "1"
    with (pasta / "execucao.log").open("w", encoding="utf-8") as log:
        for notebook in sorted(notebooks.glob("*.ipynb")):
            comando = [sys.executable, str(BASE / "scripts/executar_notebook.py"), str(notebook)]
            print(f"Executando {notebook.name}...", flush=True)
            resultado = subprocess.run(comando, cwd=BASE, env=ambiente, text=True,
                                       stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            if resultado.returncode:
                raise SystemExit(f"Falha em {notebook.name}. Confira {pasta / 'execucao.log'}.")
    from verificar_refatoracao import verificar
    verificar(nome)
    print("Execução e resultados:", pasta)


if __name__ == "__main__":
    main()
