
import numpy as np # linear algebra
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import os


iris=load_iris()
x=iris.data
y=iris.target
y_names=iris.target_names
tIds=np.random.permutation(len(x))

x_train = x[tIds[:-30]]
x_test = x[tIds[30:]]

y_train = y[tIds[:-30]]
y_test = y[tIds[30:]]

#classifying using decision tree
clf = tree.DecisionTreeClassifier()

#training (fitting) the classifier with the training set
clf.fit(x_train, y_train)

#predictions on the test dataset
pred = clf.predict(x_test)

print (pred) 
print (y_test) 
final=(accuracy_score(pred, y_test))
pre=final*100
print (pre)
