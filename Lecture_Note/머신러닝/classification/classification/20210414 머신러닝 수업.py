# 20210409 머신러닝

# 20210413 이중중첩문 문제 나옴 (기출변형)
# 20210413 class에서는 나오지 않음

import statsmodels as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import datasets
from sklearn.preprocessing import PolynomialFeatures


# 20210413 시험 관련

# github 2~4번 문제 안나옴
# 로또 번호 자동 생성기 문제 안나옴
# github 비트연산 안나옴
# github 21번 문제 안 나옴
# 패키지 쓰면 감점

# 회귀분석 복습
# 다중 공산성: 독립변수끼리 상관관계가 높으면서 나타나는 문제
# f(ax+by)=af(X)+bf(y)


# 독립변수 하나일 때, SSE
# 독립변수 두 개 일 때, LSE
# Gradient Descent에서 다음달에 한 문제 나옴

# min||y-Xb||^2 에서 값이 튀지 않도록 arg를 통해 설정
# arg min ||y-Xb||^2 + lambda||b||^2 #여기서 lambda||b||^2가 ridge이다. #L2 규제를 추가한 방식

# 경사하강법(Gradient Descent)
# 경사하강법 매우 중요
# y(hat) = x0 + w1x1 + w1x1 +...+ wnxn 을 미분을 통해 구한다.
# 경사하강법 수행 프로세스 2/N -> 개수만큼 나눴다. 별로 안 중요함.
# MSE = 1/n-2 * SSE # n-2는 자유도이므로 큰 의미는 없음

# 신뢰구간을 알 수 있을 때만 p-value 등등을 쓸 수 있다.

# 도구 불러오기
import statsmodels as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import datasets
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

# 히트맵을 통해 상관관계를 시각화 한 다음 데이터를 넣고 뺌으로써
# R-Square를 통해 증명 
# 결측치 넣고 빼고 하면서 R-square 값의 움직임을 봐야함.
# 이 작업을 반복해야함.


## df.to_csv("somename.csv")로 하면 한 번에 파일을 불러올 수 있음.


#2021 04 12 수업
# encoding  할 때 한 번에 하지 말고 각각 해야함.
# graphiz 이용하기
# https://towardsdatascience.com/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-1c50b4aa68dc


# 패키지

import graphviz


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

import pandas as pd
from sklearn.datasets import load_iris
import mglearn
import os

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X_train, X_test, Y_train, Y_test = 
train_test_split(df[data.feature_names], df['target'], random_state=0)

# Step 1: Import the model you want to use
# This was already imported earlier in the notebook so commenting out
#from sklearn.tree import DecisionTreeClassifier
# Step 2: Make an instance of the Model
clf = DecisionTreeClassifier(max_depth = 2, 
                             random_state = 0)
# Step 3: Train the model on the data
clf.fit(X_train, Y_train)
# Step 4: Predict labels of unseen (test) data
# Not doing this step in the tutorial
# clf.predict(X_test)
# Load the Breast Cancer (Diagnostic) Dataset

tree.plot_tree(clf);

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')

plt.show()


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
# Arrange Data into Features Matrix and Target Vector
X = df.loc[:, df.columns != 'target']
y = df.loc[:, 'target'].values
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
# Random Forests in `scikit-learn` (with N = 100)
rf = RandomForestClassifier(n_estimators=1000,
                            random_state=0)
rf.fit(X_train, Y_train)



import mglearn
# 가상의 그림도 그릴 수 있음

# 가상의 그림 만들기
mglearn.plots.plot_tree_progressive()
plt.show()

# meshgrid
# grid인데 3차원에서 그물망 처럼 그림

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 나 혼자 연습하기
 
data =
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
# Arrange Data into Features Matrix and Target Vector
X = df.loc[:, df.columns != 'target']
y = df.loc[:, 'target'].values
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42)
# Random Forests in `scikit-learn` (with N = 100)
rf = RandomForestClassifier(n_estimators=300,
                            random_state=100)
rf.fit(X_train, Y_train)


cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, random_state=9999)
tree= DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set:{:.3f}".format(tree.score(X_test, y_test)))


import graphviz

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

import pandas as pd
from sklearn.datasets import load_iris
import mglearn
import os
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

## KNN

# 유방암데이터로 연습

# 유방암데이터 정보
breast_cancer=load_breast_cancer()
breast_cancer.keys()
breast_cancer.target_names
breast_cancer.feature_names
breast_cancer.target
breast_cancer.data
breast_cancer.target

# 유방암데이터를 X_train, X_test, y_train, y_test로 쪼개기
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(breast_cancer.data,breast_cancer.target,
test_size = 0.2,random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# KNN 분류하기
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)

# KNN score(정확도) 보기
print('accuracy : {:.2f}'.format(knn.score(X_test,y_test)))


## Scikit-learn을 활용한 나이브 베이즈 분류기 구축
todaylunch=['rice','noodle','fastfood','rice','noodle','fastfood','rice','noodle','rice','noodle','fastfood','rice','noodle','fastfood']
foodtemp=['hot','cold','soso','hot','cold','soso','hot','cold','soso','cold','soso','hot','cold','soso']
eat=['yes','no','yes','no','yes','no','yes','no','yes','no','yes','no','no','yes']

#분류기 구축
# 오늘 점심 밥, 국수, 패스트푸드이고, 음식온도는 뜨겁거나,차갑거나,미적지근해
# 먹을겨? 말겨?

# 컴퓨터는 바보라 문자를 인식 못하니까 숫자로 바꿔주자
# label encoder로 오늘 점심이랑 음식 온도를 숫자로 바꾼뒤 저장
# 먹을지 말지도 label encoder로 숫자로 바꿔준다음에 라벨로 저장을 떄리자.

# 숫자로 다 바꿧으면 zip함수로 묶어준뒤, 리스트로 저장합시다.
# 결합까지 끝났으면 Gaussian Classifer 데리고 오자
# Gaussian Classifer로 모델을 만들어주고,  결합한 거랑, 라벨(먹을지 말지)로 모델에 넣은 다음
# 예측을 해봅시다.
# 스포주의: 결과가 1이 나왔으니, 오늘 점심은 꼭 먹도록 합시다

# label encoder를 해봅시다.
# 도구 불러오기
from sklearn import preprocessing

#label encoder 만들기
le = preprocessing.LabelEncoder()

todaylunch_encoded=le.fit_transform(todaylunch)
print(todaylunch_encoded)
#[2 1 0 2 1 0 2 1 2 1 0 2 1 0]

# converting string labels into numbers
temp_encoded=le.fit_transform(foodtemp)
label=le.fit_transform(eat)
print("FoodTemp",temp_encoded)
#FoodTemp [1 0 2 1 0 2 1 0 2 0 2 1 0 2]
print("eat",label)
#eat [1 0 1 0 1 0 1 0 1 0 1 0 0 1]

# 글자를 숫자로 전부 바꿨으니 이제 결합을 해봅시다.
# 오늘 점심 메뉴와 음식 온도를 엔코딩한 걸 zip을 이용하여 엮어보기
features=zip(todaylunch_encoded,temp_encoded)
features=list(features) #리스트화 하기
print(features)
#[(2, 1), (1, 0), (0, 2), (2, 1), (1, 0), (0, 2), (2, 1), (1, 0), (2, 2), (1, 0), (0, 2), (2, 1), (1, 0), (0, 2)]

# 결합을 다 해봤으면 이제 Gaussian Classifer 써볼까요?

# 호출!
from sklearn.naive_bayes import GaussianNB

# Gaussian Classifer 만들기!
model=GaussianNB()

# 훈련하세요~
model.fit(features,label)

#예측값 뭐나왔니?
predicted=model.predict([[0,2]])
print("Predicted Value:", predicted)
#Predicted Value: [1]


## 결정트리(Decision Tree)
# 결정트리 주요 하이퍼파라미터
# 1. max_depth #트리의 최대 깊이를 규정
# 2. max_features #최대 피처 개수
# # int: 피처의 개수, float: 피처 퍼센트, sqrt: 전체 피처중 sqrt(전체 피처 개수) 그니까 개수만큼 선정하라고
# # auto: sqrt랑 똑같이해줘 #log: 전체 피처중에 log2(전체 피처 개수)선정 
# # min_samples_spilt: 과적합 제어하는데 쓸 건데 노드를 분할하기 위한 최소한의 샘플 데이터수
# # min_samples_leaf: 말단 노드(leaf)가 되기 위한 최소한의 샘플 데이터 수(비대칭적 데이터일 경우 클래스
# 의 데이터가 극도록 작을 수 있으므로, 이 경우는 작게 설정 필요)
# max_leaf_nodes: 말단 노드(leaf)의 최대 개수

# 결정트리
# 데이터 정의
# 기술속성(features), 대상속성(target) 지정
# 엔트로피 -sigma p(x) log(p(x)) 
# -np.sum(p(X)*log2(p(X)) for i in range(len(np.unique(대상속성, 기술속성=True)))
# 정보이득 (1-엔트로피)
#  






















































