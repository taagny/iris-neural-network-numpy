"""Rede neural com NumPy para classificar as flores do Iris."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preparar_dados():
    iris = load_iris()
    entradas_treino, entradas_teste, rotulos_treino, rotulos_teste = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    # O teste usa a média e o desvio do treino para evitar vazamento de dados.
    padronizador = StandardScaler()
    entradas_treino = padronizador.fit_transform(entradas_treino)
    entradas_teste = padronizador.transform(entradas_teste)

    # Cada linha da matriz identidade representa uma das três espécies.
    saidas_treino = np.eye(3)[rotulos_treino]
    saidas_teste = np.eye(3)[rotulos_teste]
    return entradas_treino, entradas_teste, saidas_treino, saidas_teste


def relu(valores):
    return np.maximum(0, valores)


def derivada_relu(valores):
    return (valores > 0).astype(float)


def softmax(valores):
    # Subtrair o maior valor de cada linha evita estouro na exponencial.
    exponenciais = np.exp(valores - np.max(valores, axis=1, keepdims=True))
    return exponenciais / np.sum(exponenciais, axis=1, keepdims=True)


def inicializar_parametros():
    return {
        "W1": np.random.randn(4, 8) * 0.1,
        "b1": np.zeros((1, 8)),
        "W2": np.random.randn(8, 3) * 0.1,
        "b2": np.zeros((1, 3)),
    }


def propagacao_direta(entradas, parametros):
    z1 = entradas @ parametros["W1"] + parametros["b1"]
    a1 = relu(z1)
    z2 = a1 @ parametros["W2"] + parametros["b2"]
    a2 = softmax(z2)
    return z1, a1, a2


def entropia_cruzada(probabilidades, saidas):
    epsilon = 1e-10
    return -np.mean(np.sum(saidas * np.log(probabilidades + epsilon), axis=1))


def main():
    np.random.seed(42)
    entradas_treino, entradas_teste, saidas_treino, saidas_teste = preparar_dados()
    print(f"Treino: {entradas_treino.shape[0]} amostras; teste: {entradas_teste.shape[0]} amostras.")
    print(f"Rótulos one-hot: treino {saidas_treino.shape}, teste {saidas_teste.shape}.")
    parametros = inicializar_parametros()
    print(f"Pesos: W1 {parametros['W1'].shape}, W2 {parametros['W2'].shape}.")
    z1, a1, probabilidades = propagacao_direta(entradas_treino, parametros)
    perda = entropia_cruzada(probabilidades, saidas_treino)
    print(f"Loss inicial: {perda:.4f}")


if __name__ == "__main__":
    main()
