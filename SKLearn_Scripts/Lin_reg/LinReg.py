import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
import joblib

california = fetch_california_housing()

#features / labels
x=california.data
y=california.target

lin_reg = linear_model.LinearRegression()
#need to transpose x so we have the same number of dimensions
#plt.scatter(x.T[0],y)
#plt.show()


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #percentage as decimal
model = lin_reg.fit(x_train,y_train)
prediction = model.predict(x_test)
print("prediction :", prediction)
print("R^2 value :", lin_reg.score(x,y))
print("coedd: " , lin_reg.coef_)
print("intercept: ", lin_reg.intercept_)
joblib.dump(model,"./lin_reg_model.sav")