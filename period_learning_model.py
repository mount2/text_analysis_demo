import pandas as pd 
import numpy as np
import os
import json
import matplotlib 
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import normalize
import graphviz 
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import pickle

def get_score(model,x_train,x_test,y_train,y_test): 
    model.fit(x_train,y_train)
    
    return model.score(x_test,y_test)

def get_recall(model,x_train,x_test,y_train,y_test): 
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    return  recall_score(y_test,y_pred,average='weighted')

def get_precision(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    return  precision_score(y_test,y_pred,average='weighted')

df_model = pd.read_csv('concated_data.csv')
df_model = df_model.fillna(0.0)


X_COL = ['wordlen','speak_time','stop_time']
Y_COL = ['period']
score_LR = []
recall_LR = []
precision_LR =[]

kf = KFold(n_splits = 10,random_state= 55, shuffle = True)
for train_index,test_index in kf.split(df_model[Y_COL]): 
    X_train= df_model[X_COL].iloc[train_index]
    X_test = df_model[X_COL].iloc[test_index]
    y_train = df_model[Y_COL].iloc[train_index]
    y_test = df_model[Y_COL].iloc[test_index]
    model = DecisionTreeClassifier(class_weight='balanced')
    score_LR.append(get_score(model,X_train,X_test,y_train,y_test))
    recall_LR.append(get_recall(model,X_train,X_test,y_train,y_test))
    precision_LR.append(get_precision(model,X_train,X_test,y_train,y_test))






print(f'logistic regression score when use all features : ', mean(score_LR))
print(f'logistic regression recall when use all features : ', mean(recall_LR))
print(f'logistic regression precision when use all features : ', mean(precision_LR))
df_model.to_csv('concated_data.csv',index = False)