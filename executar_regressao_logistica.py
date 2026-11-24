"""Treina a Regressão Logística e a compara com a etapa de balanceamento."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from hai_tcc.classificacao.regressao_logistica import executar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanceamento", required=True, help="Nome da execução da etapa 3 em runs/.")
    parser.add_argument("--nome", help="Nome de uma pasta nova em runs/.")
    args = parser.parse_args()
    nome = args.nome or datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("regressao_logistica_%Y%m%d_%H%M%S")
    executar(RAIZ, nome, args.balanceamento)
