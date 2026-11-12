# Rede neural para classificar o Iris

Atividade acadêmica de Inteligência Artificial do curso de Ciência da Computação.
O projeto implementa uma rede neural do zero para classificar as espécies
Iris setosa, Iris versicolor e Iris virginica a partir de quatro medidas da flor.

Toda a lógica da rede está em `main.py` e usa apenas NumPy: operações matriciais,
ativações, perda, backpropagation, atualização dos pesos e cálculo da acurácia.
O Scikit-Learn é usado somente para carregar o Iris, dividir os dados e padronizar
as entradas. Não há uso de modelos prontos.

## Como instalar

No terminal, dentro da pasta do projeto, use o ambiente virtual `.venv`.
Caso ainda não exista, crie com `python -m venv .venv`.

No Windows (PowerShell), instale as dependências com:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No Linux ou macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Como rodar

No Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe main.py
```

No Linux ou macOS:

```bash
.venv/bin/python main.py
```

O Iris já acompanha o Scikit-Learn, então não é necessário baixar arquivos de dados.
O script mostra a loss e a acurácia de treino a cada 100 épocas e, ao terminar,
exibe a acurácia no conjunto de teste.

## Como a rede funciona

- Dados: 120 amostras de treino (80%) e 30 de teste (20%), com divisão estratificada
  e `random_state=42`.
- Padronização: o `StandardScaler` é ajustado apenas no treino e aplicado ao teste.
- Rótulos: one-hot de tamanho 3, montado com NumPy.
- Arquitetura: 4 entradas, 8 neurônios escondidos com ReLU e 3 saídas com softmax.
- Inicialização: pesos aleatórios multiplicados por 0,1, vieses zerados e
  `np.random.seed(42)`.
- Perda: entropia cruzada média com epsilon no logaritmo.
- Treinamento: gradiente descendente com todas as amostras de treino, 1000 épocas
  e taxa de aprendizado de 0,1.
- Avaliação: a classe prevista é o `argmax` das probabilidades. O teste é avaliado
  depois do treinamento e não participa da atualização dos pesos.

O forward pass calcula as ativações das duas camadas. O backpropagation calcula
os gradientes pela regra da cadeia, e cada parâmetro é atualizado subtraindo
a taxa de aprendizado multiplicada pelo seu gradiente.

## Resultado obtido

Com a configuração acima, a execução produziu:

```text
Época 1000 | Loss: 0.0514 | Acurácia de treino: 97.50%
Acurácia final no teste: 96.67%
```

Foram classificadas corretamente 29 das 30 amostras de teste. Esse resultado se
refere à divisão fixa usada nesta atividade.

Execução validada com Python 3.14.7, NumPy 2.5.4 e Scikit-Learn 1.9.1.
As ativações também foram verificadas com entradas grandes na softmax, e os
gradientes foram conferidos por diferenças finitas.
