"""Monta a página portátil a partir das medições e das previsões já verificadas."""

import base64
import gzip
import json
from pathlib import Path
import shutil

from dados_hai import MODELOS, SINAIS, ler_referencia, ler_sinais


RAIZ = Path(__file__).resolve().parent


def montar():
    dados = ler_referencia(ler_sinais())
    # Um byte por segundo guarda o rótulo (bit 0) e os três alertas (bits 1 a 3).
    decisoes = dados["attack_real"].to_numpy(dtype="uint8", copy=True)
    for bit, coluna in enumerate(MODELOS.values(), start=1):
        decisoes |= dados[coluna].to_numpy(dtype="uint8") << bit
    # Os sinais preservam a precisão do CSV. A primeira coluna começa no byte 0.
    blocos = [dados[sinal].to_numpy(dtype="<f8").tobytes() for sinal in SINAIS]
    pacote = gzip.compress(b"".join(blocos) + decisoes.tobytes(), compresslevel=6, mtime=0)
    meta = {
        "total": len(dados),
        "inicio": dados["time"].iloc[0].isoformat(),
        "sinais": list(SINAIS),
        "previsoes": list(MODELOS.values()),
    }
    interface = RAIZ / "interface"
    pagina = (interface / "modelo.html").read_text(encoding="utf-8")
    for marca, conteudo in (
        ("__ESTILO__", (interface / "painel.css").read_text(encoding="utf-8")),
        ("__METADADOS__", json.dumps(meta, ensure_ascii=False)),
        ("__DADOS_GZIP__", base64.b64encode(pacote).decode("ascii")),
        ("__PROGRAMA__", (interface / "painel.js").read_text(encoding="utf-8")),
    ):
        if pagina.count(marca) != 1:
            raise ValueError(f"Marcador inesperado no modelo HTML: {marca}")
        pagina = pagina.replace(marca, conteudo)
    pasta = RAIZ / "painel"
    pasta.mkdir(exist_ok=True)
    (pasta / "index.html").write_text(pagina, encoding="utf-8")
    # A imagem do cabeçalho acompanha a página portátil no mesmo diretório.
    shutil.copy2(interface / "banner_industrial.webp", pasta / "banner_industrial.webp")


if __name__ == "__main__":
    montar()
