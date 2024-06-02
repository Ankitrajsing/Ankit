

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
# print(iris.head())
df=pd.DataFrame(iris.data,columns=iris.feature_names)
# print(df)
df['target']=iris.target
# print(df.head())
input=df.drop(['target'],axis='columns')
target=df.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(input,target,test_size=0.3)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
pred=model.predict([[2.3,6.7,3.0,5.2]])
print(model.score(x_test,y_test))
print(pred)
