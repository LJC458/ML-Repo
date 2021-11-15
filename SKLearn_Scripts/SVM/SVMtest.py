#SVM uses hyperplanes for classification or regression.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn import svm
import joblib
import numpy as np
import pandas as pd

iris = datasets.load_iris()

x = iris.data
y = iris.target

classification = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #percentage as decimal

model = svm.SVC()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
acc=metrics.accuracy_score(y_test,predictions)
print(model)
print("accuracy:",acc)
joblib.dump(model,'./svm.sav')