# HAI 21.03 · Etapa 2: projeto modular

**TCC 2 de Samuel Eleoterio · 10 de novembro de 2026.**

A primeira frente sugerida pelo professor foi implementada em uma cópia do
projeto: os cinco notebooks continuam apresentando as etapas e os gráficos.
Agora eles chamam funções reutilizáveis em arquivos Python menores.

## Dashboard interativo · HAI 21.03

O projeto inclui um [dashboard interativo](Dashboard_HAI_2103/) para explorar as medições industriais e os alertas históricos do conjunto `test5`. O painel compara Regressão Logística, Isolation Forest com PCA e a combinação pela regra OU. **Não realiza detecção em tempo real**.

- [Código e arquivos do dashboard](Dashboard_HAI_2103/)
- [Documentação, funcionamento e fontes dos dados](Dashboard_HAI_2103/README.md)

**Executar localmente com Streamlit:**

```bash
cd Dashboard_HAI_2103
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Para abrir a versão HTML sem instalar Python, baixe o repositório e abra `Dashboard_HAI_2103/painel/index.html` em um navegador atualizado. O arquivo usa dados incorporados e requer suporte a `DecompressionStream`.

**Atenção às métricas:** a execução de referência incorporada ao painel apresenta 647 falsos positivos na Regressão Logística e 1.121 no ensemble; os resultados principais do artigo são 643 e 1.118, respectivamente. O [README do dashboard](Dashboard_HAI_2103/README.md) explica como importar a execução correspondente ao artigo.

## Comece por estes três pontos

1. Leia **COMECE_AQUI.md**. Ele explica o que mudou em linguagem simples.
2. Abra os cinco arquivos de **notebooks/**, em ordem. As saídas da conferência
   estão salvas. Não é necessário treinar novamente para ler os resultados.
3. Antes de executar, copie seus oito CSVs do HAI para **data/hai-21.03/**.
   A pasta não está no ZIP para manter o envio leve. O notebook 01 também pode
   baixar os arquivos ausentes da revisão registrada em **data/origem.json**.

Extraia em uma pasta separada da versão original. Os modelos e resultados
desta entrega estão em **runs/baseline_modular/**.

## O que cada parte faz

| Parte | Responsabilidade |
| --- | --- |
| notebooks/01_exploracao_hai.ipynb | Conferir os oito arquivos, os rótulos, os horários e a qualidade. |
| notebooks/02_treinamento.ipynb | Selecionar as entradas e ajustar os três modelos com o mesmo protocolo. |
| notebooks/03_validacao.ipynb | Escolher os limiares em test4, pelo F1. |
| notebooks/04_teste.ipynb | Aplicar os cortes salvos em test5 e medir os erros. |
| notebooks/05_analise_resultados.ipynb | Reunir resultados, gráficos e a síntese. |
| src/hai_tcc/data.py | Ler os arquivos, validar a estrutura e preservar a origem das linhas. |
| src/hai_tcc/features.py | Separar X e y, selecionar colunas no treino, ajustar scaler e PCA nos normais do treino. |
| src/hai_tcc/models.py | Treinar RF/IF, empacotar os modelos e calcular pontuações. |
| src/hai_tcc/evaluation.py | Escolher limiares, calcular métricas, cobertura e falsos positivos por hora. |
| src/hai_tcc/experiment.py | Resolver caminhos, carregar configurações e registrar a ordem das etapas. |
| src/hai_tcc/sampling.py | Função de subamostragem para a etapa seguinte; conferida com exemplos pequenos. |
| configs/baseline.yaml | Parâmetros explícitos da referência. |
| configs/classificacao.yaml | Plano das futuras proporções/sementes; ainda não executável. |
| tests/ | 24 testes pequenos sobre comportamento e integridade do protocolo. |
| runs/ | Uma pasta por execução, com modelos, resultados, notebooks e manifesto. |
| referencia/ | Tabelas históricas e saídas da etapa 1 para comparação. |

## Conferência realizada

As 49 células foram executadas em ordem em sessões IPython independentes por
notebook. Os oito CSVs foram conferidos por hash SHA-256. Os 24 testes passaram.

| Modelo | F1 na referência | F1 após modularizar | Limiar exibido |
| --- | ---: | ---: | ---: |
| Random Forest | 0,1855 | 0,1855 | 0,170000 |
| Isolation Forest sem PCA | 0,2203 | 0,2203 | 0,548756 |
| Isolation Forest com PCA | 0,4611 | 0,4611 | 0,567999 |

As previsões dos **92.401 registros** de test5 são idênticas às da reprodução da
etapa 1. As pontuações e os cortes coincidem com tolerância absoluta de 1e-12.
As tabelas de validação, teste, cobertura, primeiro alerta e falsos positivos por
hora também coincidem.

As 60 colunas e sua ordem continuam iguais, assim como as 19 constantes
retiradas no treino. O PCA continua com 21 componentes. A nova coluna de
metadados **indice_origem** apenas identifica a linha dentro do arquivo original;
ela não entra em X.

O log está em **runs/baseline_modular/execucao.log**. O manifesto registra a
configuração, as sementes, as versões das bibliotecas, os hashes dos módulos,
a revisão dos dados e a conclusão de cada etapa.

## Instalação e execução

O ambiente conferido usa **Python 3.12** e **scikit-learn 1.9.1**. Use uma
venv própria para esta cópia, sem alterar o ambiente do projeto anterior.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No VS Code, selecione o kernel dessa venv para abrir os notebooks.

Para rodar somente os testes pequenos:

```bash
python executar.py --testes
```

Para conferir as saídas que já estão incluídas, sem treinar:

```bash
python verificar_refatoracao.py
```

Para repetir tudo em uma pasta de execução nova:

```bash
python executar.py
```

Esse comando roda os testes, copia os notebooks, executa cada um em uma sessão
nova e compara os resultados à etapa 1. Ele cria uma pasta com data e hora em
**runs/** e não sobrescreve a referência incluída. Se uma célula falhar, o
comando para e aponta o arquivo **execucao.log**.

Para comparar manualmente uma execução com outro nome:

```bash
python verificar_refatoracao.py --experimento NOME_DA_EXECUCAO
```

Os notebooks também podem ser executados um a um, na ordem 01–05. Nesse modo,
eles usam **configs/baseline.yaml** e atualizam a pasta **baseline_modular**.
Faça isso em uma cópia se quiser preservar as saídas incluídas.

Se mudar o código ou a configuração de uma execução já registrada, crie uma
nova execução. O manifesto evita misturar parâmetros novos com saídas antigas.
Não se adiciona amostragem à validação nem ao teste.

## O que fica para a próxima etapa

A comparação real de balanceamento ainda não foi executada nesta entrega.
A função de amostragem foi preparada e testada com dados pequenos.

O plano é preservar todos os 5.613 registros de ataque do treino e comparar:
RF atual ponderado; RF completo sem pesos; RF com proporções ataque:normal
1:20, 1:10, 1:5 e 1:1. As três sementes propostas para essa comparação são
13, 42 e 2026. Elas foram definidas antes dos novos resultados.

As configurações serão escolhidas em **test4**. **Test5 será comparação
histórica**, pois já foi examinado. A análise vai incluir PR-AUC (com definição
explícita), variação entre sementes e a investigação dos intervalos 6 e 9 e
das horas com muitos falsos positivos.
