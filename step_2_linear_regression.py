# Обучение модели линейной регрессии с помощью градиентного спуска. Предсказания
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
        X.insert(0, 'intercept', 1)
        self.weights = np.ones(X.shape[1]) if self.weights is None else self.weights

        if verbose:
            y_pred = self.weights @ X.T
            loss = np.mean((y - y_pred)**2)
            print(f"start | loss: {loss:.2f}")

        for i in range(self.n_iter):
            y_pred = self.weights @ X.T
            loss = np.mean((y - y_pred)**2)
            gradient = -2 * np.mean((y - y_pred) * X.T, axis=1)
            self.weights = self.weights - self.learning_rate * gradient

            if verbose and (i+1) % verbose == 0:
                print(f"{i+1} | loss: {loss:.2f}")

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = X.copy()
        X.insert(0, 'intercept', 1)
        return self.weights @ X.T 
 
# Пример использования    
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=3, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)

model = MyLineReg(n_iter=500, learning_rate=0.01)
model.fit(X, y, verbose=100)

X_test = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
predictions = model.predict(X_test)
print(f"Сумма предсказаний: {np.sum(predictions)}") 

# Выводит:
#    start | loss: 6697.42
#    100 | loss: 153.69
#    200 | loss: 8.98
#    300 | loss: 0.66
#    400 | loss: 0.05
#    500 | loss: 0.00
#    Сумма предсказаний: 1641.6003125626194