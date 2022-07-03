import pandas as pd 
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()

x = iris.data

print(type(x))

df_model = pd.read_csv('concated_data.csv')
df_model = df_model.fillna(0.0)

X_COL = ['wordlen','speak_time','stop_time']
Y_COL = ['period']


train_x = df_model[X_COL]
train_y = df_model[Y_COL]

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()


model = LogisticRegression()
model.fit(train_x,train_y)
with open('model_pickle','wb') as f:
    pickle.dump(model,f)



with open('model_pickle','rb') as f : 
    mod = pickle.load(f)


#test = pd.DataFrame({'wordlen':10,'speak_time':0.36,'stop_time':0.0})
#print(train_x.iloc[1])

print(mod.predict([[7,0.5,3]]))
y = mod.predict(train_x)
print(y[500])
for i in y : 
    if i : 
        print(i)
    else: 
        pass