import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics

def data_clean(data, column1, column2):
    data = data.drop(columns = ["Reference"])
    X = data[[column1, column2]]
    y = data[["Actuator Type"]]
    X,y = reset(X,y,column1, column2)
    for i in range(2):
        na = X.iloc[:,i].isnull()*(-2) + 1
        to_remove = X.iloc[:,i].fillna(value = 1)*na
        for j in range(y.shape[0]):
            if to_remove[j]<0:
                X = X.drop(j)
                y = y.drop(j)
        X,y = reset(X,y,column1, column2)
    X = X.to_numpy()
    X = np.log(X)
    mean = np.mean(X,axis = 0)
    #std = np.std(X,axis = 0)
    std = 1
    X = (X-mean)/std
    return X,y.to_numpy().reshape(y.shape[0])

def reset(X,y,column1, column2):
    X = X.reset_index()
    y = y.reset_index()
    X = X[[column1, column2]]
    y = y[["Actuator Type"]]
    return X,y

def plot_multilabel_boundary(X,y,linear, rbf, poly, sig,column1,column2):
    h = .01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']

    for i, clf in enumerate((linear, rbf, poly, sig)):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4) 
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        y_map = {'PZT':0, 'DEA':1, 'IPMC':2, 'SMA':3, 'SFA':4, 'TSA':5, 'SCP':6, 'EAP':7, 'SMP':8}
        for j in range(len(Z)):
            Z[j] = y_map[Z[j]]
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn,     edgecolors='grey')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        plt.show()