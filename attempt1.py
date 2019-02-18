# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:43:18 2017

@author: Khagendra


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import Imputer


#Reading the data
""" Common in all modules"""
dataset=pd.read_csv('C:\Users\shrey\Desktop\Project Mortality progress\file1.csv')
X=dataset.iloc[:,9:-1].values
Y=dataset.iloc[0:50,39].values


#Missing Data
"""
Created on Sun Oct 15 18:47:17 2017

@author: Khagendra

The following fills the missing data
"""
imputer=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer=imputer.fit(X[:,9:38])
X[0:50,9:38]=imputer.transform(X[0:50,9:38])


#Splitting test and training
"""
Created on Sun Oct 15 18:52:45 2017

@author: Khagendra
The following section splits the training and test sections
"""
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


#Feature Scaling
"""
Created on Sun Oct 15 18:54:53 2017

@author: Khagendra

The following section does feature scaling.Feature scaling is used so that entries with extremely large values like the
deaths in the year 1955 does not dominate the other values.
"""


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

sc_Y=StandardScaler()
Y_train=sc_Y.fit_transform(Y_train)
Y_test=sc_Y.transform(Y_test)



#Regression Tree 
""" Presently incomplete"""



from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(criterion="mae",random_state=0)
regressor.fit(X_train,Y_train)

pred=[]
for i in range(9,38):
    k=int(input("Entries to be the values to be predicted:"))
    pred.append(k)
    
Y_pred=regressor.predict(pred)


#plot
""" Plotting the graph"""
plt.scatter(X,Y,color="red")
plt.plot(X,regressor.predict(X),color="blue")
plt.title('Mortality Rate Graph')
plt.show()


"""Saving the model"""

import pickle
s=pickle.dumps(regressor)


"""Loading the model"""
regressor1=pickle.load(s)
""" Using the model"""
#regressor1.predict(pred)


