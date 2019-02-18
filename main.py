# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:00:35 2017

@author: Khagendra
"""

import pandas as pd
import Missing_data as MD
import Splitting_data as SD
import Feature_Scaling as FS
import regrTree as rt

#Reading the data
""" Common in all modules"""
dataset=pd.read_csv('file1.csv')
X=dataset.iloc[:,9:-1].values
Y=dataset.iloc[0:50,39].values
X=MD.MisDat(X)
X_train,X_test,Y_train,Y_test=SD.splitdat(X,Y)
X_train,X_test,Y_train,Y_test=FS.scale()
s=rt.regr(X_train,Y_train,X,Y)



import pickle
"""Loading the model"""
regressor1=pickle.load(s)
""" Using the model"""
regressor1.predict(X)
