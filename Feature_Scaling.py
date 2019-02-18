# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:54:53 2017

@author: Khagendra

The following section does feature scaling.Feature scaling is used so that entries with extremely large values like the
deaths in the year 1955 does not dominate the other values.
"""


from numba import jit

@jit
def scale(X_train,X_test,Y_train,Y_test):
    
    
    
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)
    
    sc_Y=StandardScaler()
    Y_train=sc_Y.fit_transform(X_train)
    Y_test=sc_Y.transform(X_test)
    
    return X_train,X_test,Y_train,Y_test
    
    