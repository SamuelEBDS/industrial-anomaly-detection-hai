# O que fizemos nesta etapa?

Imagine que os notebooks eram cinco cadernos e que algumas receitas estavam
repetidas dentro deles. Agora as receitas ficam em arquivos próprios. O caderno
continua mostrando o que você fez, mas chama a receita pronta.

Por exemplo, no notebook 03:

```python
from hai_tcc.evaluation import selecionar_limiar
```

Essa linha busca a função no arquivo **src/hai_tcc/evaluation.py**. Quando
chamamos a função com os rótulos e as pontuações de test4, ela calcula qual
corte maximiza o F1. O critério continua o mesmo que você já explicou.

## Macete: caderno, receita e ficha

- **Notebook = caderno:** conta a história e mostra tabelas e gráficos.
- **Módulo Python = receita:** guarda uma operação que você pode repetir.
- **Configuração YAML = ficha:** anota quais arquivos e parâmetros usar.
- **Teste de código = conferência:** usa um exemplo pequeno cuja resposta é conhecida.
- **Pasta runs = registro do experimento:** guarda o que foi executado e produzido.

O “teste de código” é diferente de **test5**. O primeiro verifica o programa com
exemplos pequenos; o segundo mede os modelos nos dados reservados do HAI.

## Em que ordem estudar?

1. Abra o notebook 01. Veja a chamada que lê e confere os arquivos.
2. Abra **src/hai_tcc/data.py** e procure **ler_particao**. Ela abre os CSVs,
   guarda a origem e ordena as linhas no tempo.
3. Abra o notebook 02. Veja como **separar_entradas** coloca as medições em X,
   o rótulo attack em y e o horário/origem nos metadados.
4. Depois, veja **features.py**: as constantes são escolhidas no treino, e o
   scaler/PCA aprendem nos registros normais desse treino.
5. Nos notebooks 03 e 04, procure **pontuar** e **aplicar_limiar**. Pontuar
   usa modelos e transformações já ajustados. Aplicar o limiar gera 0 ou 1.
6. O notebook 05 continua reunindo e explicando as saídas.

## Como sabemos que a reorganização funcionou?

Não olhamos só se apareceu “sem erro”. Comparamos cada previsão de test5 à
versão anterior, além dos cortes, pontuações e tabelas. As previsões ficaram
iguais. Os F1 continuam 0,1855, 0,2203 e 0,4611.

Também usamos 24 testes pequenos. Um deles coloca um número extremo na
validação e verifica que ele não muda a média que o scaler aprendeu no treino.
Outro verifica se um alarme fora de um trecho de ataque é separado da detecção
daquele trecho.

## Uma frase para explicar ao professor

“Mantive os cinco notebooks e extraí as operações para módulos reutilizáveis.
Conferi a mudança com testes e reproduzi as mesmas previsões e métricas.
Agora posso comparar formas de balancear o treino sem duplicar o código.”

## A próxima pergunta da pesquisa

“Se eu mantiver os ataques e usar menos exemplos normais no treino do Random
Forest, consigo detectar melhor sem gerar muitos falsos alarmes?”

Essa rodada de experimentos vem a seguir; ainda não há resultados novos de
balanceamento neste pacote.
