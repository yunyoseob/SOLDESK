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
print("hello")

# 1. 도구들 불러오기
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

# 2. 파일불러오기
df_sampleSubmission=pd.read_csv('C:/Users/sundooedu/Desktop/bike-sharing-demand/sampleSubmission.csv')
df_test=pd.read_csv('C:/Users/sundooedu/Desktop/bike-sharing-demand/test.csv')
df_train=pd.read_csv('C:/Users/sundooedu/Desktop/bike-sharing-demand/train.csv')


# 3. 데이터 구성 보기
df_train.info()
df_train_1=df_train.copy()
df_test_1=df_test.copy()

df_train_1['datetime']=pd.to_datetime(df_train_1['datetime'])
df_train_1.dtypes

df_train_1.isna().sum()

df_train_1['datetime']=pd.to_datetime(train["datetime"]) 
#판다스가 인식하는 Datetime 타입과 서식타입의 오차를 사전에 방지하는 줄

df_train_1['year'] = df_train_1['datetime'].dt.year
df_train_1['month'] = df_train_1['datetime'].dt.month
df_train_1['day'] = df_train_1['datetime'].dt.day
df_train_1['hour'] = df_train_1['datetime'].dt.hour
df_train_1['minute'] = df_train_1['datetime'].dt.minute
df_train_1['second'] = df_train_1['datetime'].dt.second

#요일 데이터 -일요일은 0
df_train_1['dayofweek'] = df_train_1['datetime'].dt.dayofweek

df_train_1.describe()

# 차트 보기
figure, ((ax1, ax2, ax3),(ax4, ax5,ax6))=plt.subplots(nrows=2,ncols=3)
figure.set_size_inches(18,8)

sns.barplot(data=df_train_1, x="year", y="count", ax=ax1)
sns.barplot(data=df_train_1, x="month", y="count", ax=ax2)
sns.barplot(data=df_train_1, x="day", y="count", ax=ax3)
sns.barplot(data=df_train_1, x="hour", y="count", ax=ax4)
sns.barplot(data=df_train_1, x="minute", y="count", ax=ax5)
sns.barplot(data=df_train_1, x="second", y="count", ax=ax6)

ax1.set(ylabel='Count', title ="Year rental amount")
ax2.set(ylabel='month', title ="Month rental amount")
ax3.set(ylabel='day', title ="Day rental amount")
ax4.set(ylabel='hour', title ="Hour rental amount")

plt.show() #차트를 보여주세요

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12,10)
sns.boxplot(data=df_train_1, y="count", orient= "v", ax=axes[0][0])
sns.boxplot(data=df_train_1, y="count", x = "season",orient= "v", ax=axes[0][1])
sns.boxplot(data=df_train_1, y="count", x="hour",orient= "v", ax=axes[1][0])
sns.boxplot(data=df_train_1, y="count", x="workingday",orient= "v", ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Rental amount")
axes[0][1].set(xlabel='Season',ylabel='Count',title="Seasonal Rental amount")
axes[1][0].set(xlabel='Hour of The Day',ylabel='Count',title="Hour Rental amount")
axes[1][1].set(xlabel='Working Day',ylabel='Count',title="Working or not Rental amount")

plt.show()

fig, (ax1,ax2,ax3,ax4,ax5)=plt.subplots(nrows=5)
fig.set_size_inches(18,25)

#꺽은선 그래프.
sns.pointplot(data=df_train_1, x="hour",y="count",ax=ax1)

sns.pointplot(data=df_train_1, x="hour",y="count", hue="workingday",ax=ax2)

sns.pointplot(data=df_train_1, x="hour",y="count", hue="dayofweek",ax=ax3)

sns.pointplot(data=df_train_1, x="hour",y="count", hue="weather",ax=ax4)

sns.pointplot(data=df_train_1, x="hour",y="count", hue="season",ax=ax5)

plt.show()

# corrMatt = train[["temp","atemp","casual","registered","humidity","windspeed","count"]]
corrMatt = df_train_1.corr()
print(corrMatt)
mask =np.array(corrMatt)
#Return the indices for the upper-triangle of arr.
mask[np.tril_indices_from(mask)]=False

# 히트맵 보기
fig,ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8,square= True, annot=True)
plt.show()

#산점도로 보기
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)
fig.set_size_inches(12,5)
sns.regplot(x="temp",y="count",data=df_train_1, ax=ax1)
sns.regplot(x="windspeed",y="count",data=df_train_1, ax=ax2)
sns.regplot(x="humidity",y="count",data=df_train_1, ax=ax3)
plt.show()

#월별 데이터 모아보기
def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)
df_train_1["year_month"] = df_train_1["datetime"].apply(concatenate_year_month)
print(df_train_1.shape)
df_train_1[["datetime", "year_month"]].head()

fig, (ax1, ax2) =plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(18,4)

sns.barplot(data=df_train_1, x="year",y="count",ax=ax1)
sns.barplot(data=df_train_1, x="month",y="count",ax=ax2)

fig, ax3 = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(18,4)

sns.barplot(data=df_train_1, x="year_month",y="count",ax=ax3)

plt.show()

# 이상치 처리
# IQR(Interquartile Range)=Q3-Q1
# 1.5IQR = 분위수 범위를 확장시킴
# Q1-1.5*(Q3-Q1): 최소제한선
# Q3+1.5*(Q3-Q1): 최대 제한선

# 'count' 데이터에서 전체의 25%에 해당하는 데이터 조회
count_q1 = np.percentile(df_train_1['count'], 25)
count_q1

# 'count' 데이터에서 전체의 75%에 해당하는 데이터 조회
count_q3 = np.percentile(df_train_1['count'], 75)
count_q3

# IQR = Q3 - Q1
count_IQR = count_q3 - count_q1
count_IQR

# 이상치를 제외한(이상치가 아닌 구간에 있는) 데이터만 조회
df_train_1_IQR = df_train_1[(df_train_1['count'] >= (count_q1 - (1.5 * count_IQR))) & (df_train_1['count'] <= (count_q3 + (1.5 * count_IQR)))]
print(df_train_1_IQR)

# 3-sigma, 평균 +-3*표준편차차
df_train_1_sigma = df_train_1[np.abs(df_train_1["count"] - df_train_1["count"].mean()) <= (3*df_train_1["count"].std())]
print(df_train_1_sigma)

#IQR을 적용했을 때의 그림
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12,10)
sns.boxplot(data=df_train_1_IQR, y="count", orient= "v", ax=axes[0][0])
sns.boxplot(data=df_train_1_IQR, y="count", x = "season",orient= "v", ax=axes[0][1])
sns.boxplot(data=df_train_1_IQR, y="count", x="hour",orient= "v", ax=axes[1][0])
sns.boxplot(data=df_train_1_IQR, y="count", x="workingday",orient= "v", ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Rental amount")
axes[0][1].set(xlabel='Season',ylabel='Count',title="Seasonal Rental amount")
axes[1][0].set(xlabel='Hour of The Day',ylabel='Count',title="Hour Rental amount")
axes[1][1].set(xlabel='Working Day',ylabel='Count',title="Working or not Rental amount")

plt.show()

# 3-sigma
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12,10)
sns.boxplot(data=df_train_1_sigma, y="count", orient= "v", ax=axes[0][0])
sns.boxplot(data=df_train_1_sigma, y="count", x = "season",orient= "v", ax=axes[0][1])
sns.boxplot(data=df_train_1_sigma, y="count", x="hour",orient= "v", ax=axes[1][0])
sns.boxplot(data=df_train_1_sigma, y="count", x="workingday",orient= "v", ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Rental amount")
axes[0][1].set(xlabel='Season',ylabel='Count',title="Seasonal Rental amount")
axes[1][0].set(xlabel='Hour of The Day',ylabel='Count',title="Hour Rental amount")
axes[1][1].set(xlabel='Working Day',ylabel='Count',title="Working or not Rental amount")

plt.show()

# target과 features 구분
# target = 'count'
features = df_train_1_IQR.drop(['count'],axis = 1)

X = df_train_1_IQR.copy()
X.drop('year_month',axis=1)
y = df_train_1_IQR['count'].copy()

X_test = test.copy()

# 데이터를 편리하게 분할해주는 라이브러리 활용
from sklearn.model_selection import train_test_split

# 훈련 데이터의 25%를 검증 데이터로 활용
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)

X_test.describe()

# make_scorer 는 score랑 error를 동시에 찍을 수 있다.
from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values):
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다. # log뒤에 숫자가 1미만을 방지하기 위하여
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)
    
    # 평균을 낸다.
    mean_difference = difference.mean()
    
    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)
    
    return score

rmsle_scorer = make_scorer(rmsle)
rmsle_scorer

make_scorer(rmsle)

## 로지스틱회귀분석
## Odds
## 위스콘신 유방암 예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

type(cancer)
dir(cancer)
cancer.data.shape
cancer.feature_names
cancer.target_names
cancer.target
np.bincount(cancer.target)
print(cancer.DESCR)

for i,name in enumerate(cancer.feature_names):
      print('%02d : %s' %(i,name))

print('data =>',cancer.data.shape)
print('target =>',cancer.target.shape)

malignant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]

print('malignant(악성) =>',malignant.shape)
print('benign(양성) =>',benign.shape)

_, bins=np.histogram(cancer.data[:,0], bins=20)
np.histogram(cancer.data[:,0], bins=20)

plt.hist(malignant[:,0],bins=bins, alpha=0.3)
plt.show()

plt.hist(benign[:,0], bins=bins ,alpha=0.3)
plt.show()

plt.title(cancer.feature_names[0])
plt.show()

##
plt.figure(figsize=[20,15])

for col in range(30):
    plt.subplot(8,4,col+1)
    _, bins=np.histogram(cancer.data[:,col], bins=20)

    plt.hist(malignant[:,col],bins=bins, alpha=0.3)
    plt.hist(benign[:,col], bins=bins ,alpha=0.3)
    plt.title(cancer.feature_names[col])
    if col==0: plt.legend(cancer.target_names)
    plt.xticks([])

plt.show()

##
from sklearn.linear_model import LogisticRegression

scores = []

for i in range(10):
    X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size=0.2,random_state = 777)

    model = LogisticRegression(max_iter = 5000)
    model.fit(X_train,y_train)

    score = model.score(X_test,y_test)
    scores.append(score)

print('scores =', scores)

##
fig=plt.figure(figsize=[14,14])
fig.suptitle('Breast Cancer - feature analysis', fontsize=20)

for col in range(cancer.feature_names.shape[0]): # 30 features
    plt.subplot(8,4,col+1)
    _,bins=np.histogram(cancer.data[:,col],bins=50)
    plt.hist(malignant[:,col], bins=bins, alpha=0.5, label='malignant', color='red')
    plt.hist(benign[:,col], bins=bins, alpha=0.5, label='benign', color='green')
    
    plt.title(cancer.feature_names[col]+('(%d)' % col))
    plt.xticks([])
    plt.yticks([])
    if col==0: plt.legend()

plt.show()

##로지스틱 돌린 결과
fig=plt.figure(figsize=[14,14])
fig.suptitle('Breast Cancer - feature analysis', fontsize=20)

for col in range(cancer.feature_names.shape[0]): # 30 features
    plt.subplot(8,4,col+1)
#     f_,bins=np.histogram(cancer.data[:,col],bins=50)
#     plt.hist(malignant[:,col], bins=bins, alpha=0.5, label='malignant', color='red')
#     plt.hist(benign[:,col], bins=bins, alpha=0.5, label='benign', color='green')
    plt.scatter(cancer.data[:,col], cancer.target, c=cancer.target, alpha=0.5)
    
    
    plt.title(cancer.feature_names[col]+('(%d)' % col))
    plt.xticks([])
    plt.yticks([])
#     if col==0: plt.legend()

plt.show()

##끝














































