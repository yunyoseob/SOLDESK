# otto-group 문제풀이

#불러오기
sampleSubmission=pd.read_csv('C:/Users/sundooedu/Desktop/otto-group-product-classification-challenge/sampleSubmission.csv')
test=pd.read_csv('C:/Users/sundooedu/Desktop/otto-group-product-classification-challenge/test.csv')
train=pd.read_csv('C:/Users/sundooedu/Desktop/otto-group-product-classification-challenge/train.csv')

#도구
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import tree
import pandas as pd
import mglearn
import os
from sklearn import preprocessing
import seaborn as sns
import datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from collections import Counter
from sklearn.metrics import classification_report


#특징보기
train.describe()
train.isna().sum()
train.head()
train.shape()
train.values
train.shape
train.columns
train.target
train.keys()
train.info()
test.info()

#데이터 프레임 변환
df_train=pd.DataFrame(train)
df_train

df_train.describe()

#이상치 확인
from collections import Counter 

def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
        
        return multiple_outliers
        
detect_outliers(df_train, 2, "df_train") #키-에러

df_train.loc[Outliers_to_drop]


#데이터 전처리
#Box plot
type(df_train)


#상관계수 구하기
corr=df_train.corr(method='pearson')
print(corr)

#상관계수 시각화
corr_df_train=df_train.corr().round(2)
sns.heatmap(data=corr_df_train, annot=True)
plt.show()

# X_train, X_test, y_train, y_test로 분류하기
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train.data,train.target,
test_size = 0.2,random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# KNN 스코어 보기
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
print('accuracy : {:.2f}'.format(knn.score(X_test,y_test)))
# accuracy: 0.93

#데이터 프레임 변환 후 시각화
df_X_train=pd.DataFrame(X_train)
df_X_train

#df_X_train
corr_df_X_train=df_X_train.corr().round(2)
sns.heatmap(data=corr_df_X_train, annot=True)
plt.show()

#df_X_test
df_X_test=pd.DataFrame(X_test)
df_X_test

corr_df_X_test=df_X_test.corr().round(2)
sns.heatmap(data=corr_df_X_test, annot=True)
plt.show()

#df_y_train
df_y_train=pd.DataFrame(y_train)
df_y_train

#df_y_test
df_y_test=pd.DataFrame(y_test)
df_y_test

print(df_X_train.shape)
print(df_X_test.shape)
print(df_y_train.shape)
print(df_y_test.shape)



# 이상치 확인, 데이터 전처리(0,1), 라벨링 등등





