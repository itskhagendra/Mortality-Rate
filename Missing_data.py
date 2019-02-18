# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:47:17 2017

@author: Khagendra

The following fills the missing data
"""

from sklearn.preprocessing import Imputer
from numba import jit
@jit
def MisDat(X):   
    #Filling the most frequent value in place of empty values
    imputer=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
    imputer=imputer.fit(X[:,9:38])
    X[:,9:38]=imputer.transform(X[:,9:38])
    
    return X