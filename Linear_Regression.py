# ------------------- #

import numpy as np

class Linear_Regression():

    def __init__(self):
        self.weights = None
        self.X = None

    def fit_gd(self, X, y, init_weights=False, learning_rate=1e-01, epochs=100):
        m = len(X)

        self.X = np.concatenate([np.ones(m).reshape(-1,1), X], axis=1)

        if init_weights:
            self.weights = init_weights.copy()
        else:
            self.weights = np.zeros(self.X.shape[1])

        history = {
            'loss':[],
            'gradient':[],
            'weights':[],
            'y_pred':[]
        }

        for _ in range(epochs):
            y_pred = self.X.dot(self.weights)
            error = y_pred - y
            gradient = 2 * self.X.T.dot(error) / m
            self.weights -= learning_rate * gradient

            history['loss'].append(error.T.dot(error) / m)
            history['gradient'].append(gradient)
            history['weights'].append(self.weights)
            history['y_pred'].append(y_pred)

        return history
    
    def fit(self, X, y):
        m = len(X)
        self.X = np.concatenate([np.ones(m).reshape(-1,1), X], axis=1)
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        m = len(X)
        self.X = np.concatenate([np.ones(m).reshape(-1,1), X], axis=1)
        return self.X.dot(self.weights.T)

# ------------------- #