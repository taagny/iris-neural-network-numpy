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


def retropropagacao(entradas, saidas, z1, a1, a2, parametros):
    quantidade_amostras = entradas.shape[0]

    # A derivada da softmax com entropia cruzada simplifica para (A2 - Y) / m.
    dz2 = (a2 - saidas) / quantidade_amostras
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ parametros["W2"].T
    dz1 = da1 * derivada_relu(z1)
    dw1 = entradas.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}


def atualizar_parametros(parametros, gradientes, taxa_aprendizado):
    for nome in parametros:
        parametros[nome] -= taxa_aprendizado * gradientes[nome]


def main():
    np.random.seed(42)
    taxa_aprendizado = 0.1
    quantidade_epocas = 1000

    entradas_treino, entradas_teste, saidas_treino, saidas_teste = preparar_dados()
    print(
        f"Treino: {entradas_treino.shape[0]} amostras; "
        f"teste: {entradas_teste.shape[0]} amostras."
    )
    parametros = inicializar_parametros()

    # Cada atualização usa todas as amostras de treino (gradiente descendente).
    for epoca in range(1, quantidade_epocas + 1):
        z1, a1, probabilidades = propagacao_direta(entradas_treino, parametros)
        gradientes = retropropagacao(
            entradas_treino, saidas_treino, z1, a1, probabilidades, parametros
        )
        atualizar_parametros(parametros, gradientes, taxa_aprendizado)

        if epoca % 100 == 0:
            # O log mostra o resultado depois da atualização desta época.
            _, _, probabilidades = propagacao_direta(entradas_treino, parametros)
            perda = entropia_cruzada(probabilidades, saidas_treino)
            previsoes = np.argmax(probabilidades, axis=1)
            rotulos = np.argmax(saidas_treino, axis=1)
            acuracia = np.mean(previsoes == rotulos)
            print(
                f"Época {epoca:4d} | Loss: {perda:.4f} | "
                f"Acurácia de treino: {acuracia * 100:.2f}%"
            )

    # O conjunto de teste só é usado para avaliar a rede após o treinamento.
    _, _, probabilidades_teste = propagacao_direta(entradas_teste, parametros)
    previsoes_teste = np.argmax(probabilidades_teste, axis=1)
    rotulos_teste = np.argmax(saidas_teste, axis=1)
    acuracia_teste = np.mean(previsoes_teste == rotulos_teste)
    print(f"\nAcurácia final no teste: {acuracia_teste * 100:.2f}%")


if __name__ == "__main__":
    main()
