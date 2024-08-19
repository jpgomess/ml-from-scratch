# ------- K NEAREST NEIGHBORS ------- #

import numpy as np

from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def euclidean_distance(self, x):
        return np.sum((x - self.X_train)**2, axis=1)

    def predict(self, X):
        y_pred = []
        for x in X:
            distance = self.euclidean_distance(x)

            nearest_idx = np.argsort(distance)[:self.k]
            nearest_y_tain = self.y_train[nearest_idx]

            y_pred.append(Counter(nearest_y_tain).most_common(1)[0][0])
        return np.asarray(y_pred)

# ------- K NEAREST NEIGHBORS ------- #