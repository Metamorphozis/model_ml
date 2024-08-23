import pandas as pd
import numpy as np

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, weights={self.weights}"

    def fit(self, X, y, verbose=False):
        X = X.copy()
        X.insert(0, 'intercept', 1)  # Добавляется фиктивная фича (единичный столбец) в начало X (с помощью insert(0, 'intercept', 1))
        self.weights = np.ones(X.shape[1]) if self.weights is None else self.weights  # Инициализируем веса единицами или используем заданные

        if verbose:
            y_pred = self.weights @ X.T
            loss = np.mean((y - y_pred)**2)
            print(f"start | loss: {loss:.2f}")

        for i in range(self.n_iter):
            y_pred = self.weights @ X.T # Вычисляется предсказание y_pred
            loss = np.mean((y - y_pred)**2) # Вычисляется ошибка loss (MSE)
            gradient = -2 * np.mean((y - y_pred) * X.T, axis=1) # Вычисляется градиент ошибки
            self.weights = self.weights - self.learning_rate * gradient # Обновляются веса, используя градиент и скорость обучения

            if verbose and (i+1) % verbose == 0:
                print(f"{i+1} | loss: {loss:.2f}") # Выводится лог, если verbose не равно False

    # Возвращает веса модели, начиная со второго элемента (исключая свободный член)
    def get_coef(self):
        return self.weights[1:]
    
# Пример использования
X = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]})
y = pd.Series([3, 7, 11, 15, 19])

model = MyLineReg(n_iter=300, learning_rate=0.01)
model.fit(X, y, verbose=100) 
coef = model.get_coef()
print(f"Коэффициенты модели: {coef}") 
# Выведет:
# start | loss: 3.00
# 100 | loss: 0.37
# 200 | loss: 0.18
# 300 | loss: 0.09
# Коэффициенты модели: x1    1.162258
# x2    1.324516
# dtype: float64