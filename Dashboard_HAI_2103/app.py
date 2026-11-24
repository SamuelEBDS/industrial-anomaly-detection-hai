"""Abre a interface HTML do dashboard HAI 21.03, sem margens do Streamlit.

A interface, os dados e a lógica de avaliação continuam em /painel.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as componentes

RAIZ = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Análise de detecção · HAI 21.03",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# A faixa clara de ~16px é do contêiner exterior do Streamlit, e não do
# gráfico HTML. Zeramos o espaçamento entre o bloco CSS e o componente.
st.markdown("""
<style>
:root, html, body, #root, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #111820 !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stHeader"],
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
#MainMenu, footer { display: none !important; height: 0 !important; }

.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    max-width: none !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stElementContainer"],
[data-testid="stMarkdown"],
[data-testid="stMarkdownContainer"],
[data-testid="stCustomComponentV1"] {
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* O st.markdown que injeta o CSS não deve reservar uma linha vazia. */
[data-testid="stMarkdown"]:has(style),
[data-testid="stElementContainer"]:has([data-testid="stMarkdown"] style) {
    min-height: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

iframe[title*="painel_hai"],
iframe[title*="painel"] {
    display: block !important;
    width: 100% !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

painel = componentes.declare_component("painel_hai", path=str(RAIZ / "painel"))
painel(key="painel_hai_historico", default=None)
