
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

dataset=pd.read_csv('file2.csv')
print("File Loaded")
X=dataset.iloc[:,9:-1].values
Y=dataset.iloc[:,39].values

print("Data Loaded")
imputer=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer=imputer.fit(X[:,9:38])
X[:,9:38]=imputer.transform(X[:,9:38])

print("Data Completed")
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print("Training set:")
print("test sets:")

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
print("Transform done")

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(criterion="mae",random_state=0)
regressor.fit(X_train,Y_train)
print("Regression done")



Y_pred=regressor.predict(X_test)
print("Predicted")

from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy:",acc)

from sklearn.externals import joblib
joblib.dump(regressor, '75000.pkl')
