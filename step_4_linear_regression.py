# Использование разных типов регуляризации (L1, L2, Elasticnet)
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, weights={self.weights}, metric={self.metric}, reg={self.reg}, l1_coef={self.l1_coef}, l2_coef={self.l2_coef}"

    def fit(self, X, y, verbose=False):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        self.weights = np.ones(X.shape[1]) if self.weights is None else self.weights

        if verbose:
            y_pred = self.weights @ X.T
            loss = self._calculate_loss(y, y_pred)
            print(f"start | loss: {loss:.2f}")

        for i in range(self.n_iter):
            y_pred = self.weights @ X.T
            loss = self._calculate_loss(y, y_pred)
            gradient = self._calculate_gradient(X, y, y_pred)
            self.weights = self.weights - self.learning_rate * gradient

            if verbose and (i+1) % verbose == 0:
                print(f"{i+1} | loss: {loss:.2f}")

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        return self.weights @ X.T 

    # Вычисление потерь с учетом регуляризации
    def _calculate_loss(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred)**2)
        if self.reg == "l1":
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        elif self.reg == "l2":
            loss += self.l2_coef * np.sum(self.weights**2)
        elif self.reg == "elasticnet":
            loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights**2)
        return loss

    # Вычисление градиента с учетом регуляризации
    def _calculate_gradient(self, X, y, y_pred):
        gradient = -2 * np.mean((y - y_pred) * X.T, axis=1)
        if self.reg == "l1":
            gradient += self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            gradient += 2 * self.l2_coef * self.weights
        elif self.reg == "elasticnet":
            gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        return gradient

# Проверка. Для каждого типа регуляризации создается модель и обучается с соответствующими параметрами регуляризации.
# Выводится сумма коэффициентов обученной модели.
regs = [None, "l1", "l2", "elasticnet"]
for reg in regs:
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    if reg == "l1":
        model = MyLineReg(n_iter=300, learning_rate=0.01, reg=reg, l1_coef=0.1)
    elif reg == "l2":
        model = MyLineReg(n_iter=300, learning_rate=0.01, reg=reg, l2_coef=0.1)
    elif reg == "elasticnet":
        model = MyLineReg(n_iter=300, learning_rate=0.01, reg=reg, l1_coef=0.1, l2_coef=0.1)
    else:
        model = MyLineReg(n_iter=300, learning_rate=0.01)
    model.fit(X, y, verbose=100)
    coef = model.get_coef()
    print(f"Регуляризация {reg}: {np.sum(coef):.2f}")
    
# Вывод:
#    start | loss: 19064.15
#    100 | loss: 656.52
#    200 | loss: 36.72
#    300 | loss: 2.41
#    Регуляризация None: 313.69
#    start | loss: 19064.75
#    100 | loss: 687.03
#    200 | loss: 68.29
#    300 | loss: 34.04
#    Регуляризация l1: 313.41
#    start | loss: 19064.75
#    100 | loss: 2358.11
#    200 | loss: 1987.56
#    300 | loss: 1973.96
#    Регуляризация l2: 282.01
#    start | loss: 19065.35
#    100 | loss: 2385.86
#    200 | loss: 2015.94
#   300 | loss: 2002.36
#    Регуляризация elasticnet: 281.76