# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:52:45 2017

@author: Khagendra
The following section splits the training and test sections
"""




from numba import jit

@jit
def splitdat(X,Y): 
    
    
    #Splitting test and training
    from sklearn.cross_validation import train_test_split
    return train_test_split(X,Y,test_size=0.2,random_state=0)
