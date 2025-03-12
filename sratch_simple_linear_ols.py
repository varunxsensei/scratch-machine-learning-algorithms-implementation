import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LinearRegressionOLS:
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self,X_train,y_train):
        y_mean = np.mean(y_train)
        X_mean = np.mean(X_train)
        num = np.sum((y_train-y_mean)*(X_train-X_mean))
        den = np.sum((X_train-X_mean)**2)
        self.m = num/den
        self.b = y_mean - (self.m*X_mean)
        print(f'''value of slope m:- {self.m}
                  value of bais b:- {self.b}''')
    def predict(self,X_test):
        return X_test*self.m + self.b
    
def main():
    placement_data = pd.read_csv('./placement.csv')

    X = placement_data.iloc[:,0].values
    y = placement_data.iloc[:,1].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    reg = LinearRegressionOLS()

    reg.fit(X_train,y_train)

    reg.predict(X_test)

if __name__ == "__main__":
    main()    