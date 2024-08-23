import pandas as pd
import numpy as np
import random

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None # Инициализируем веса как None
        self.metric = metric
        self.best_score = None
        self.last_metric = 'mae'
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X, y, verbose=False):
        # Фиксируем сид для воспроизводимости
        random.seed(self.random_state)

        # Используем np.c_ для добавления единичного столбца в X
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.zeros(X.shape[1])

        for i in range(1, self.n_iter + 1):
            if self.sgd_sample is not None:
                # Выборка мини-пакета
                sample_size = round(self.sgd_sample * X.shape[0]) if isinstance(self.sgd_sample, float) else self.sgd_sample
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
                X_sample = X[sample_rows_idx]
                y_sample = np.take(y, sample_rows_idx)
            else:
                X_sample = X
                y_sample = y

            # Расчет градиента на основе мини-пакета
            y_pred = np.dot(X_sample, self.weights)
            gradient = 2 * np.dot(X_sample.T, (y_pred - y_sample)) / X_sample.shape[0]

            # Регуляризация градиента
            if self.reg == "l1":
                gradient += self.l1_coef * np.sign(self.weights)
            elif self.reg == "l2":
                gradient += 2 * self.l2_coef * self.weights
            elif self.reg == "elasticnet":
                gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            # Вычисляем learning_rate
            current_learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate

            self.weights -= current_learning_rate * gradient

            # Расчет ошибки и метрики на всем датасете
            y_pred_all = np.dot(X, self.weights)
            loss = ((y - y_pred_all)**2).mean()

            score = self._calculate_metric(y, y_pred_all)

            if verbose or i == self.n_iter:
                print(f"Iteration {i} | Loss: {loss:.2f} | Learning Rate: {current_learning_rate:.2f} | {self.metric}: {score:.2f}")

            if i == self.n_iter:
                self.best_score = score
                self.last_metric = self.metric

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        # Используем np.c_ для добавления единичного столбца в X
        X = np.c_[np.ones(X.shape[0]), X]
        predictions = np.dot(X, self.weights)
        return predictions

    def get_best_score(self):
        if self.best_score is None:
            return "No score available"
        else:
            return round(self.best_score, 10)

    def _calculate_metric(self, y_true, y_pred):
        if self.metric is None:
            self.metric = self.last_metric

        if self.metric == 'mae':
            score = np.abs(y_true - y_pred).mean()
        elif self.metric == 'mse':
            score = ((y_true - y_pred)**2).mean()
        elif self.metric == 'rmse':
            score = np.sqrt(((y_true - y_pred)**2).mean())
        elif self.metric == 'mape':
            score = np.abs((y_true - y_pred) / y_true).mean() * 100
        elif self.metric == 'r2':
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            score = 1 - (ss_res / ss_tot)
        else:
            raise ValueError("Unsupported metric")

        return score

# Тестовые данные
# Пример использования
X = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]})
y = np.array([3, 7, 11, 15, 19])

# Создаем модель
model = MyLineReg(n_iter=500, learning_rate=0.01)

# Обучаем модель 
model.fit(X, y, verbose=100)

# Получаем коэффициенты модели 
coef = model.get_coef()

print(model) # Вывод: MyLineReg class: n_iter=500, learning_rate=0.01, weights=[..., ..., ...]
print(f"Коэффициенты модели: {coef}") 

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
line = MyLineReg(n_iter=100, learning_rate=0.1, metric='mae', sgd_sample=0.1)
line.fit(X, y, verbose=False)
print(sum(line.get_coef()))
