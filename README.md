# Classificação do Iris com uma rede neural do zero

**Atividade acadêmica de Inteligência Artificial · Ciência da Computação**

Uma rede neural implementada em Python para classificar três espécies de flores
a partir de quatro medidas. O projeto acompanha todo o processo de aprendizado:
preparação dos dados, propagação direta, cálculo da perda, backpropagation e
atualização dos pesos com gradiente descendente.

Toda a lógica da rede usa **NumPy** e está em um único arquivo, [`main.py`](main.py).
O **Scikit-Learn** é utilizado exclusivamente para carregar e preparar os dados.

> **Resultado obtido:** 96,67% de acurácia no teste — 29 classificações corretas
> em 30 amostras, após 1000 épocas de treinamento.

## Objetivo da atividade

Compreender como uma rede neural aprende sem utilizar modelos prontos. A
implementação explicita as operações matriciais e a regra da cadeia usadas para
ajustar os parâmetros de uma rede com uma camada escondida.

## Dataset e preparação dos dados

O Iris é carregado por `load_iris`, sem necessidade de baixar arquivos separados.
São 150 amostras, distribuídas igualmente entre três espécies.

| Entradas da rede | Unidade |
| --- | --- |
| Comprimento da sépala | cm |
| Largura da sépala | cm |
| Comprimento da pétala | cm |
| Largura da pétala | cm |

| Classe | Espécie | Representação one-hot |
| --- | --- | --- |
| 0 | Iris setosa | `[1, 0, 0]` |
| 1 | Iris versicolor | `[0, 1, 0]` |
| 2 | Iris virginica | `[0, 0, 1]` |

A preparação segue esta sequência:

1. Divisão estratificada em **80% para treino** e **20% para teste**, com
   `random_state=42`, preservando a proporção das classes.
2. Ajuste do `StandardScaler` somente nas entradas de treino, para obter média
   zero e desvio padrão um em cada característica desse conjunto.
3. Transformação das entradas de teste com as estatísticas calculadas no treino,
   evitando vazamento de dados.
4. Conversão dos rótulos para one-hot com `np.eye(3)`.

| Conjunto | Amostras | Amostras por espécie | Formato das entradas | Formato dos rótulos |
| --- | --- | --- | --- | --- |
| Treino | 120 | 40 | `120 × 4` | `120 × 3` |
| Teste | 30 | 10 | `30 × 4` | `30 × 3` |

## Arquitetura da rede

```mermaid
flowchart LR
    E["Entrada: 4 características"] --> H["Camada escondida: 8 neurônios · ReLU"]
    H --> S["Saída: 3 neurônios · Softmax"]
    S --> C["Classe prevista: argmax"]
```

| Parâmetro | Dimensão | Inicialização |
| --- | --- | --- |
| `W1` | `4 × 8` | `np.random.randn(4, 8) * 0.1` |
| `b1` | `1 × 8` | Zeros |
| `W2` | `8 × 3` | `np.random.randn(8, 3) * 0.1` |
| `b2` | `1 × 3` | Zeros |

A rede possui **67 parâmetros treináveis**. A inicialização usa
`np.random.seed(42)` para permitir a reprodução do experimento.

## Como o aprendizado acontece

### 1. Propagação direta

As entradas percorrem as duas camadas:

```text
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2
A2 = Softmax(Z2)
```

A ReLU retorna `max(0, x)`. Sua derivada vale 1 para valores positivos e 0
para valores menores ou iguais a zero, conforme implementado no projeto.

A softmax transforma as saídas em probabilidades cuja soma é 1 em cada amostra.
Antes de calcular as exponenciais, o maior valor de cada linha é subtraído para
evitar estouro numérico.

### 2. Cálculo da perda

A entropia cruzada mede a diferença entre as probabilidades previstas e os
rótulos esperados:

```text
Loss = -soma(Y * log(A2 + epsilon)) / m
```

`m` é o número de amostras e `epsilon = 1e-10` evita o cálculo de `log(0)`.
A perda é calculada como a média entre as amostras.

### 3. Backpropagation

Os gradientes são calculados da saída para a entrada, pela regra da cadeia:

```text
dZ2 = (A2 - Y) / m
dW2 = A1.T @ dZ2
db2 = soma de dZ2 por coluna
dA1 = dZ2 @ W2.T
dZ1 = dA1 * ReLU'(Z1)
dW1 = X.T @ dZ1
db1 = soma de dZ1 por coluna
```

As somas dos gradientes dos vieses mantêm as dimensões `1 × 3` e `1 × 8`.

### 4. Atualização dos parâmetros

Cada época utiliza todas as amostras de treino para calcular os gradientes.
Pesos e vieses são atualizados por gradiente descendente:

```text
parâmetro = parâmetro - taxa_aprendizado * gradiente
```

| Configuração | Valor |
| --- | --- |
| Épocas | 1000 |
| Taxa de aprendizado | 0,1 |
| Amostras por atualização | 120, todo o conjunto de treino |
| Intervalo dos logs | A cada 100 épocas |
| Seed do NumPy | 42 |
| Estado aleatório da divisão | 42 |

Os logs mostram a perda e a acurácia de treino após a atualização da época.
Ao final, a rede faz uma propagação direta no teste e escolhe a classe de maior
probabilidade com `argmax`.

## Instalação e execução

É necessário ter Python 3 e as dependências de [`requirements.txt`](requirements.txt).
Execute os comandos a partir da pasta do projeto. Se a `.venv` já existir,
pule o comando de criação do ambiente.

### Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### Linux ou macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Os comandos usam diretamente o Python do ambiente virtual, sem precisar ativá-lo.
No PyCharm, selecione o interpretador da `.venv` do projeto e execute `main.py`.

## Resultados

Trecho da saída obtida com a configuração padrão:

```text
Treino: 120 amostras; teste: 30 amostras.
Época  100 | Loss: 0.3821 | Acurácia de treino: 86.67%
Época  200 | Loss: 0.2007 | Acurácia de treino: 92.50%
Época  300 | Loss: 0.1248 | Acurácia de treino: 95.83%
...
Época 1000 | Loss: 0.0514 | Acurácia de treino: 97.50%

Acurácia final no teste: 96.67%
```

| Métrica | Resultado |
| --- | --- |
| Loss de treino após 1000 épocas | 0,0514 |
| Acurácia de treino | 97,50% — 117 de 120 amostras |
| Acurácia de teste | **96,67% — 29 de 30 amostras** |

A acurácia é calculada com NumPy pela proporção de previsões corretas. O
conjunto de teste não participa do ajuste do padronizador nem da atualização
dos pesos.

O resultado corresponde a uma única divisão fixa do Iris. Não foi realizada
validação cruzada; portanto, a acurácia apresentada descreve esse experimento.

## Organização do projeto

```text
ia-rede-neural-iris-numpy/
├── main.py           Implementação completa e execução do experimento
├── requirements.txt  Dependências: NumPy e Scikit-Learn
├── .gitignore        Exclusão da .venv, .idea e arquivos de cache
└── README.md         Documentação da atividade
```

## Verificações realizadas

- Conferência das dimensões das entradas, rótulos, pesos e vieses.
- Verificação da padronização e da distribuição das classes nos conjuntos.
- Conferência da ReLU, de sua derivada e da estabilidade numérica da softmax.
- Comparação dos gradientes do backpropagation com diferenças finitas.
- Execuções repetidas para conferir a reprodutibilidade dos resultados.

O experimento foi executado com **Python 3.14.7**, **NumPy 2.5.4** e
**Scikit-Learn 1.9.1**. O `requirements.txt` lista as dependências sem fixar versões;
para reproduzir o ambiente validado, utilize essas mesmas versões.
