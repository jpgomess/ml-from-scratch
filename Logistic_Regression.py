# ------- LOGISTIC REGRESSION ------- #

import numpy as np

class Logistic_Regression:

    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, X, y_true, y_pred):
        m = len(X)
        dw = (1 / m) * np.dot(X.T, (y_pred - y_true))
        db = (1 / m) * np.sum(y_pred - y_true)
        return dw, db

    def fit(self, X, y, learning_rate=1e-3, epochs=100, print_rate=10):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        losses = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            dw, db = self.backward(X, y, y_pred)
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            losses.append(loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss {loss:.4f}')

        history = {'epochs' : np.arange(epochs) + 1,'loss' : np.asarray(losses)}
        return history

    def predict(self, X):
        y_pred = self.forward(X)
        y_pred_class = np.where(y_pred > 0.5, 1, 0)
        return y_pred_class

    def predict_proba(self, X):
        y_pred = self.forward(X)
        return y_pred

# ------- LOGISTIC REGRESSION ------- #