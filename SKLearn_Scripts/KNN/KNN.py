import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

data = pd.read_csv('car.data')
print(data.head())

X = data[[
         'buying'
        ,'maint'
        ,'safety'
]].values

y = data[['class']]

print(X,y)

 # convert labels into encodings

Le = LabelEncoder()
for i in range(len(X[0])):
     X[:,i] = Le.fit_transform(X[:,i])

print(X)

label_mapping ={
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}

y['class']=y['class'].map(label_mapping)
y=np.array(y)
print(y)

#Create model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #percentage as decimal

knn_model = knn.fit(x_train,y_train)
dump(knn_model,'.\knn.sav')
#prediction =knn.predict(x_test)
#
#accuracy=metrics.accuracy_score(y_test, prediction)
#
#print('prediciton: ', prediction)
#print('accuracy ', accuracy)
#
#print('actual value :' ,y_test[20])
#print('predicted value :',prediction[20])

