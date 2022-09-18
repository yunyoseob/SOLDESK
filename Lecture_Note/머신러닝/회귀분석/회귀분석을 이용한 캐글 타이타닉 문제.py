#회귀분석을 이용한 캐글 타이타닉 문제
# 순서
# 1. 도구 불러오기
# 2. 문제파일 불러오기
# 3. 각종 통계 및 자료 확인
# 4. 함수 정의
# 5. 차트 보기
# 6. dataset 설정
# 7. one-hot encoding
# 8. Missing data 처리
# 9. # 9. 변수의 분포를 시각화 및 여러 변수들 사이의 상관관계를 보고 싶다.
# sns.FacetGrid(A, hue='B', aspect=C)

#변수의 분포를 시각화하거나, 여러 변수들 사이의 상관관계를 여러개의 그래프로 쪼개서 표현할때 유용함
# FeactGrid는 Colum,row, hue를 통한 의미구분을 통해 총 3차원까지 구현이 가능함.
#aspect : subplot의 세로 대비 가로의 비율.
# 끝

# 1. 도구 불러오기
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
import seaborn as sns

# 2. 캐글 타이타닉 문제 파일 불러오기
test=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/test.csv')  
test

train=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/train.csv')
train

gender_submission=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/gender_submission.csv')
gender_submissionp

# 3. 각종 통계 및 자료 확인
test.describe() #기술통계량 요약
train.describe()
train.head() # 위에 10개만
test.head()
train.info() # 구성
test.info()
print(train.shape,test.shape) #행, 열 개수
train.isna().sum() # 널 값의 개수

# 범주형에 대해 feature에 대한 막대 차트
# Pclass, sex, sibsp, parch, embarked, cabin

# 4. 함수 정의
def bar_chart(feature):
     survived=train[train['Survived']==1][feature].value_counts() #생존자를 카운트
     dead=train[train['Survived']==0][feature].value_counts() #사망자를 카운트
     df=pd.DataFrame([survived,dead]) #[생존자, 사망자]를 dataFrame
     df.index=['Survived','Dead'] #index화
     df.plot(kind='bar',stacked=True, figsize=(10,5)) #그림을 그림


#함수 설명
# survived: 생존=1, 죽음=0
# pclass: 승객 등급. 1등급=1, 2등급=2, 3등급=3
# sibsp: 함께 탑승한 형제 또는 배우자 수
# parch: 함께 탑승한 부모 또는 자녀 수
# ticket: 티켓 번호
# cabin: 선실 번호
# embarked: 탑승장소 S=Southhampton, C=Cherbourg, Q=Queenstown

# 5. 차트 보기
bar_chart('Sex')
plt.show()

bar_chart('SibSp')
plt.show()

bar_chart('Parch')
plt.show()

bar_chart('Age')
plt.show()

# 6. dataset 설정
train_test_data = [train,test]
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
# '([A-Za-z]+)\.' A-Z,a-z 순서대로 나열 expand=False

train['Title'].value_counts()
train[train['Survived']==1]['Sex']

# 7. one-hot encoding
# 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여,
# 다른 인덱스에는 0을 부여하는 단어의 벡터표현 방식
# 각 단어에 고유한 인덱스를 부여한 후 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고,
# 다른 단어의 인덱스의 위치에는 0을 부여합니다.

title_mapping = {'Mr':0, "Miss":1, 'Mrs':2,'Master':3,
                 'Dr':3,'Rev':3,'Col':3,'Major':3,'Mlle':3,'Ms':3,'Sir':3,'Don':3,'Countess':3,
                 'Capt':3,'Lady':3,'Jonkheer':3,'Mme':3}
for dataset in train_test_data:
  dataset['Title'] = dataset['Title'].map(title_mapping)

bar_chart('Title')
plt.show()

## 이런식으로 고유값을 부여

## 원래 전부 one-hot encoding을 해야하지만, 너무 귀찮은 관계로 drop함수를 이용하여 드랍
train.drop('Name',axis = 1, inplace = True)
test.drop('Name',axis = 1, inplace = True)
train.drop('Ticket',axis = 1, inplace = True)
test.drop('Ticket',axis = 1, inplace = True)
train.drop('Cabin',axis = 1, inplace = True)
test.drop('Cabin',axis = 1, inplace = True)
train.drop('Embarked',axis = 1, inplace = True)
test.drop('Embarked',axis = 1, inplace = True)

train.head()
test.head()

## mapping 함수를 이용하여 datset 형성
sex_mapping = {'male':0,'female':1}
for dataset in train_test_data:
  dataset['Sex'] = dataset['Sex'].map(sex_mapping)

bar_chart('Sex')
plt.show()

# 8. Missing Data 
train.isna().sum() #Age null값이 177개 있음
test.isna().sum()  #Age null값이 86개 있음

train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace = True)
# inplace=True는 고정해서 저장하라는 의미

test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace = True)
test['Fare'].fillna(test.groupby('Title')['Fare'].transform('median'),inplace = True)
test['Title'].fillna(0,inplace = True)

# 9. 변수의 분포를 시각화 및 여러 변수들 사이의 상관관계를 보고 싶다.
# sns.FacetGrid(A, hue='B', aspect=C)

#변수의 분포를 시각화하거나, 여러 변수들 사이의 상관관계를 여러개의 그래프로 쪼개서 표현할때 유용함
# FeactGrid는 Colum,row, hue를 통한 의미구분을 통해 총 3차원까지 구현이 가능함.
#aspect : subplot의 세로 대비 가로의 비율.

# 참고
# def bar_chart(feature):
#     survived=train[train['Survived']==1][feature].value_counts() #생존자를 카운트
#     dead=train[train['Survived']==0][feature].value_counts() #사망자를 카운트
#     df=pd.DataFrame([survived,dead]) #[생존자, 사망자]를 dataFrame
#     df.index=['Survived','Dead'] #index화
#     df.plot(kind='bar',stacked=True, figsize=(10,5)) #그림을 그림

#함수 설명
# survived: 생존=1, 죽음=0
# pclass: 승객 등급. 1등급=1, 2등급=2, 3등급=3
# sibsp: 함께 탑승한 형제 또는 배우자 수
# parch: 함께 탑승한 부모 또는 자녀 수
# ticket: 티켓 번호
# cabin: 선실 번호
# embarked: 탑승장소 S=Southhampton, C=Cherbourg, Q=Queenstown



## 연령별로 산사람과 죽은 사람을 보고 싶을 때

facet = sns.FacetGrid(train, hue ='Survived', aspect=4)
facet.map(sns.kdeplot,'Age',shade = True) # kde : 이차원 밀집도 그래프
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
sns.axes_style('dark')

plt.show()

##
facet = sns.FacetGrid(train, hue ='Survived', aspect=4)
facet.map(sns.kdeplot,'Age',shade = True) # kde : 이차원 밀집도 그래프
facet.set(xlim=(0,train['Age'].max()))  #x는 0에서 ~까지만을 xlim= 을 통해 표현
facet.add_legend()
sns.axes_style('dark')

plt.xlim(0,20)

plt.show()

## 함께 동승한 부모님과 아이들의 수화 형제와 배우자 수
# sibsp: 함께 탑승한 형제 또는 배우자 수
# parch: 함께 탑승한 부모 또는 자녀 수
# 혼자 탄 사람이 있을 거고, 가족들이랑 탄 사람이 있을 건데, 생존률이 어떻게 다를까?

train['FamilySize'] = train['SibSp']+train['Parch']+1
# train['Sibsp']랑 train['Parch']를 결합을 해줌.

test['FamilySize'] = test['SibSp']+test['Parch']+1

#결합이 끝났으니 표로 비교해서 보자
facet = sns.FacetGrid(train2, hue ='Survived', aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade = True) # kde : 이차원 밀집도 그래프
facet.set(xlim=(0,train2['FamilySize'].max()))
facet.add_legend()

plt.show()

# 혼자일 경우 사망률이 높을까?
X_train = train.drop(['Survived','PassengerId'],axis=1)
Y_train = train['Survived']
X_test = test.drop('PassengerId',axis = 1).copy()

X_train
Y_train
X_test


































































