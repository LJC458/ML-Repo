
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot  as plt
import joblib


bc = load_breast_cancer()

x = scale(bc.data)
y = bc.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #percentage as decimal

model = KMeans(n_clusters=2, random_state=0)

model.fit(x_train)

predictions = model.predict(x_test)

labels=model.labels_

accuracy = metrics.accuracy_score(y_test, predictions)

joblib.dump(model, "./KMeans.sav")