"""Compara o Random Forest com subamostragem e a referência já conferida."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from hai_tcc.classificacao.experimento import executar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nome", help="Nome de uma pasta nova em runs/.")
    parser.add_argument("--referencia", default="baseline_modular", help="Execução modular completa em runs/.")
    args = parser.parse_args()
    nome = args.nome or datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("balanceamento_rf_%Y%m%d_%H%M%S")
    executar(RAIZ, nome, args.referencia)
