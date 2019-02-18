# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:52:30 2017

@author: Khagendra
"""

from numba import jit

@jit
def regr(X_train,Y_train,X,Y):
    from sklearn.tree import DecisionTreeRegressor
    regressor=DecisionTreeRegressor(criterion="mae",random_state=0)
    regressor.fit(X_train,Y_train)
    
    pred=[]
    for i in range(9,38):
        k=int(input("Entries to be the values to be predicted:"))
        pred.append(k)
        
    Y_pred=regressor.predict(pred)
    print("Future possible value:"+Y_pred)
    return Y_pred
    
    #plot
    """ Plotting the graph"""


    import matplotlib.pyplot as plt
    plt.scatter(X,Y,color="red")
    plt.plot(X,regressor.predict(X),color="blue")
    plt.title('Mortality Rate Graph')
    plt.show()
    
    """Saving the model"""

    import pickle
    s=pickle.dumps(regressor)
    
    return s
