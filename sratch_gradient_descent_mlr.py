import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

class GDRegressor:
    def __init__(self,learning_rate,epochs):
       self.lr = learning_rate
       self.epochs = epochs
       self.weights = None
    def fit(self,X_train,y_train):
        m,n = X_train.shape
        ones = np.ones((m,1))
        X_train_trf = np.concatenate((ones,X_train),axis = 1)

        self.weights = np.random.rand(X_train_trf.shape[1])

        for i in range(self.epochs):
            y_pred = np.dot(X_train_trf,self.weights)
            weight_slope = -2 * np.dot(X_train_trf.T, (y_train - y_pred)) 
            self.weights = self.weights - (self.lr*weight_slope)

        intercept_ = self.weights[0]
        coef_ = self.weights[1:]
        print(f"coef_ :- {coef_} and intercept_ :- {intercept_}")

    def predict(self,X_test):
        m = X_test.shape[0]
        ones = np.ones((m, 1))
        X_test_trf = np.concatenate((ones, X_test), axis=1)
        return np.dot(X_test_trf, self.weights)   