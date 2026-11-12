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


def main():
    np.random.seed(42)
    entradas_treino, entradas_teste, saidas_treino, saidas_teste = preparar_dados()
    print(f"Treino: {entradas_treino.shape[0]} amostras; teste: {entradas_teste.shape[0]} amostras.")
    print(f"Rótulos one-hot: treino {saidas_treino.shape}, teste {saidas_teste.shape}.")


if __name__ == "__main__":
    main()
