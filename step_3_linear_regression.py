# Рассчет метрик
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, weights={self.weights}, metric={self.metric}"

    def fit(self, X, y, verbose=False):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        self.weights = np.ones(X.shape[1]) if self.weights is None else self.weights

        if verbose:
            y_pred = self.weights @ X.T
            loss = np.mean((y - y_pred)**2)
            self.best_score = self._calculate_metric(y, y_pred)
            print(f"start | loss: {loss:.2f} | {self.metric}: {self.best_score:.2f}")

        for i in range(self.n_iter):
            y_pred = self.weights @ X.T
            loss = np.mean((y - y_pred)**2)
            gradient = -2 * np.mean((y - y_pred) * X.T, axis=1)
            self.weights = self.weights - self.learning_rate * gradient

            if verbose and (i+1) % verbose == 0:
                self.best_score = self._calculate_metric(y, y_pred)
                print(f"{i+1} | loss: {loss:.2f} | {self.metric}: {self.best_score:.2f}")

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        return self.weights @ X.T 

    # Возвращает последнее значение метрики
    def get_best_score(self):
        return self.best_score

    # Вычисление выбранной метрики
    def _calculate_metric(self, y_true, y_pred):
        if self.metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.metric == 'mse':
            return np.mean((y_true - y_pred)**2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred)**2))
        elif self.metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif self.metric == 'r2':
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)
        else:
            return None

# Проверка. В коде проверки добавлен цикл по всем доступным метрикам
# Для каждой метрики создается модель и обучается.
# Выводится значение метрики после обучения модели
metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
for metric in metrics:
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = MyLineReg(n_iter=300, learning_rate=0.01, metric=metric)
    model.fit(X, y, verbose=100)
    best_score = model.get_best_score()
    print(f"Метрика {metric}: {best_score:.2f}")
    
# Вывод:
#    start | loss: 19064.15 | mae: 109.87
#    100 | loss: 656.52 | mae: 20.83
#    200 | loss: 36.72 | mae: 4.94
#    300 | loss: 2.41 | mae: 1.26
#    Метрика mae: 1.26
#    start | loss: 19064.15 | mse: 19064.15
#    100 | loss: 656.52 | mse: 656.52
#    200 | loss: 36.72 | mse: 36.72
#    300 | loss: 2.41 | mse: 2.41
#    Метрика mse: 2.41
#    start | loss: 19064.15 | rmse: 138.07
#    100 | loss: 656.52 | rmse: 25.62
#    200 | loss: 36.72 | rmse: 6.06
#    300 | loss: 2.41 | rmse: 1.55
#    Метрика rmse: 1.55
#    start | loss: 19064.15 | mape: 126.31
#    100 | loss: 656.52 | mape: 471.64
#    200 | loss: 36.72 | mape: 175.24
#    300 | loss: 2.41 | mape: 53.16
#    Метрика mape: 53.16
#    start | loss: 19064.15 | r2: 0.03
#    100 | loss: 656.52 | r2: 0.97
#    200 | loss: 36.72 | r2: 1.00
#    300 | loss: 2.41 | r2: 1.00
#    Метрика r2: 1.00
    