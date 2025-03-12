import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

class MultipleLinearRegressionOLS:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self,X_train,y_train):
        ones = np.ones((X_train.shape[0],1))
        X_train_trf = np.concatenate((ones,X_train),axis = 1)
        weights = np.linalg.inv(np.dot(X_train_trf.T,X_train_trf)).dot(np.dot(X_train_trf.T,y_train))
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        print("Model fitted...")

    def predict(self,X_test):
        return X_test*self.coef_ + self.intercept_

def main():
    X,y = load_diabetes(return_X_y=True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    lr = MultipleLinearRegressionOLS()
    
    lr.fit(X_train,y_train)

    print(lr.coef_)

    print(lr.intercept_)

if __name__ == "__main__":
    main()    