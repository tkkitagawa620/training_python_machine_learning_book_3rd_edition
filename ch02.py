import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np


class Perceptron(object):
    """パーセプトロン分類機

    パラメータ
    ------------
    eta : float
      学習率（0.0より大きく1.0以下）
    n_iter : int
      訓練データの訓練回数
    random_state : int
      重みを初期化するための乱数シード

    属性
    -----------
    w_ : 1d-array
      適合後の重み
    errors_ : list
      各エポックでの誤分類（更新）の数

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """訓練データに適合させる

        パラメータ
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          n_examplesは訓練データの個数
          n_featuresは特徴量の個数
        y : array-like, shape = [n_examples]
          目的変数

        返り値
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """単位ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_desicion_regions(x, y, classifier, resolution=0.02):
    markers = ('s', 's', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors=('black')
                    )


def exctract_iris_dataset():
    df = pd.read_csv('python-machine-learning-book-3rd-edition/ch02/iris.data', header=None)
    df.tail()

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    x = df.iloc[0:100, [0, 2]].values

    plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='o', label='versicolor')
    plt.xlabel('sepal length [cm')
    plt.ylabel('petal length [cm')
    plt.legend(loc="upper left")
    # plt.show()

    return (y, x)


if __name__ == "__main__":
    y, x = exctract_iris_dataset()
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(x, y)
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of updates')
    # plt.show()
    plot_desicion_regions(x, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.show()
