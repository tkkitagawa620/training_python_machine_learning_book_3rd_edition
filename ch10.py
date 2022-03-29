from turtle import color
from sklearn.preprocessing import StandardScaler
import numpy as np
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('python-machine-learning-book-3rd-edition/ch10/housing.data.txt', header=None, sep='\\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
# scatterplotmatrix(df[cols].values, figsize=(10, 8), names=cols, alpha=0.5)
# plt.tight_layout()
# plt.show()

# cm = np.corrcoef(df[cols].values.T)
# hm = heatmap(cm, row_names=cols, column_names=cols)
# plt.show()


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)
plt.plot(range(1, lr.n_iter + 1), lr.cost_)
plt.xlabel('ITER')
plt.ylabel('COST')
plt.show()


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolors='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None


lin_regplot(X_std, y_std, lr)
plt.xlabel('average number of rooms[RM](standardized)')
plt.ylabel('Prince in $1000s [MEDV](standardized))')
plt.show()
