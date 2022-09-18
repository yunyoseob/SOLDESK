# 20210407 머신러닝 회귀분석 수업
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

# 다항식으로 변환한 단항식 생성, [[0,1],[2,3]]의 2*2 행렬 생성
X=np.arange(4).reshape(2,2)
print('일차 단항식 계수 feature:\n', X)

# degree = 2인 2차 다항식으로 변환하기 위해 PolynomialFeatures를 이용하여 변환
poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr=poly.transform(X)
print('변환된 2차 다항식 계수 feature:/n', poly_ftr)

test=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/test.csv')
test

train=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/train.csv')
train

gender_submission=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/gender_submission.csv')
gender_submission

train.describe()

# 문제? 연속형? 범주형?

def bar_chart(feature):
     survived=train[train['Survived']==1][feature].values_counts() #생존자를 카운트
     dead=train[train['Survived']==0][feature].values_counts() #사망자를 카운트
     df=pd.DataFrame([survived,dead]) #[생존자, 사망자]를 dataFrame
     df.index=['Survived','Dead'] #index화
     df.plot(kind='bar',stacked=True, figsize=(10,5)) #그림을 그림

train_test_data = [train,test]
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False) 
# [A-Za-z] A~Z까지 a~z까지 \. 정규표현식
# ([A-Za-z]+)\.  A~Z까지 a~z까지 정규표현식으로 나타내라

train['Title'].value_counts()

#  캐글 타이타닉 문제는 주피터 20210407 머신러닝 파일에 저장되어있음.

for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=26),'Age'] = 1
    dataset.loc[(dataset['Age']>26)&(dataset['Age']<=36),'Age'] = 2
    dataset.loc[(dataset['Age']>36)&(dataset['Age']<=46),'Age'] = 3
    dataset.loc[dataset['Age']>46,'Age'] = 4

print(train['Age'])
print(test['Age'])


# 20210408 회귀분석 머신러닝














































