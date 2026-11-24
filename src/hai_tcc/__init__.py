"""Funções reutilizáveis dos cinco notebooks do TCC com HAI 21.03."""

MODELOS = ("RF", "IF sem PCA", "IF com PCA")
SUFIXOS = {"RF": "rf", "IF sem PCA": "if_sem_pca", "IF com PCA": "if_com_pca"}
ARQUIVOS_MODELOS = {
    "RF": "random_forest_referencia.joblib",
    "IF sem PCA": "isolation_forest_sem_pca.joblib",
    "IF com PCA": "isolation_forest_com_pca.joblib",
}
