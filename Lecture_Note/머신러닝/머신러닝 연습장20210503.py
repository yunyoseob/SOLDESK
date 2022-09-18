# 머신러닝 연습장
print('hi')

# iris 데이터
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


iris=load_iris()
iris

type(iris.data) #numpy.ndarray
type(iris.target) #numpy.ndarray
iris.data # (150, 4) 행렬인 float64 튜플
iris.data.dtype # dtype('float64')
iris.data.size # 600
iris.data.ndim # 2차원
iris.data.shape # 150,4 행렬

iris.target # 0,1,2로 이루어져있는  int 튜플
iris.target.dtype # dtype('int32')
iris.target.size #150
iris.target.ndim # 1차원
iris.target.shape # (150,) 행렬

iris.keys()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])


type(iris.feature_names) # iris의 칼럼: class 'list'
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

type(iris.target_names) # iris의 칼럼 numpy.ndarray
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')


# train데이터에 넣을 것과 test데이터에 넣을 때, numpy 행렬의 행이 일치해야 함.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.2,random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#load_wine 데이터
from sklearn import datasets

wine=datasets.load_wine()

wine.feature_names # wine데이터의 index들
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
# 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
# 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

wine.data
wine.data.shape #(178, 13)
wine.target.size #(178)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Accuracy: 0.9074074074074074


#
# 유방암 데이터로 살펴보는 Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

cancer.keys()
# dict_keys(['data', 'target', 'frame', 
# 'target_names', 'DESCR', 'feature_names', 'filename'])

cancer.data.shape #(569,30)
cancer.target.shape #(569,)



X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42) # stratify : target:
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# Accuracy on training set: 1.000
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
# Accuracy on test set: 0.937

# 의사결정트리 만들기
tree = DecisionTreeClassifier(max_depth=10, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
#Accuracy on training set: 1.000
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
#Accuracy on test set: 0.937

type(cancer.feature_names) # numpy.ndarray
cancer.feature_names.size


import mglearn
import matplotlib as plt
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

iris.keys()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

dataframe=pd.DataFrame(iris.data, columns=iris.features_names)
scatter_matrix(dataframe, c=iris.target, marker='o', s=10, alpha=.8, figsize(12,8))
plt.show()

#
cancer.keys()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

import mglearn
mglearn.plots.plot_scaling()
plt.show()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
print(X_train.shape) # 426, 30
print(X_test.shape) # 143, 30










































































