import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

class GDRegressor:
    def __init__(self,learning_rate,epochs):
        self.epochs = epochs
        self.lr = learning_rate
        self.coef_ = np.random.randn()
        self.intercept_ = np.random.randn()
    
    def fit(self,X_train,y_train):
        for i in range(self.epochs):
            y_pred = self.coef_*X_train + self.intercept_
            coef_slope = -2*np.mean((y_train-y_pred)*X_train)
            intercept_slope = -2*np.mean((y_train-y_pred))
            self.coef_ = self.coef_ - ( self.lr*coef_slope)
            self.intercept_ = self.intercept_ - (self.lr*intercept_slope)

        print(f'''coefs_ :- {self.coef_}
                 intercept:- {self.intercept_}''')

    def predict(self,X_test):
        return self.coef_ * X_test + self.intercept_


def main():
    df = pd.read_csv('./placement.csv')
    X = df.iloc[:,0].values
    y = df.iloc[:,1].values
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    gdr = GDRegressor(learning_rate=0.01,epochs=1000)

    gdr.fit(X_train,y_train)


if __name__ == "__main__":
    main()
