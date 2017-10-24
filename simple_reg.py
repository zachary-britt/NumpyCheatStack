import numpy as np
import numpy.linalg as npl
import scipy.stats as scs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston




class LogisticRegression:

    def __init__(self):
        pass



class LinearRegression:

    def __init__(self,X,y,add_constant=True,regularization=None):
        '''
        Input:  X (n, m) matrix
                y (n, 1) column vector
                add_constant (add ones column to X)
                regularization (None, ridge, or lasso)
        '''
        self.X = X

        self.n, self.m = X.shape
        self.p = self.m
        n1 = len(y)
        self.y = y.reshape(n1,1)

        if self.n != n1:
            print('X must have the same number of rows as y')
            exit(1)

        if add_constant:
            self.add_constant_to_X()

        self.beta = npl.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(y)
        self.y_hat = self.X.dot(self.beta)
        self.residuals = self.y_hat - self.y


    def normalize_params(self):
        pass


    def add_constant_to_X(self):
        self.m += 1
        X_1 = np.ones(shape=(self.n,self.m))
        X_1[:,1:] = self.X
        self.X = X_1


    def info(self):
        TSS = sum((self.y-self.y.mean())**2)
        RSS = sum(self.residuals**2)
        RSE = (RSS/(self.n-self.p-1))**0.5
        Rsqrd = (TSS-RSS)/TSS


        RMSE = (RSS/self.n)**0.5
        s=""
        s += "RMSE:\t{}".format(RMSE)+'\n'
        s += "RSS:\t{}".format(RSS)+'\n'
        s += "RSE:\t{}".format(RSE)+'\n'
        s += "Rsqrd:\t{}".format(Rsqrd)+'\n'
        return s


if __name__ == '__main__':
    boston = load_boston()
    X = boston.data # housing features
    y = boston.target # housing prices

    model=LinearRegression(X,y)
    #print(model.info())
    print(model.n)











    #beta
