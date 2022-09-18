print("Hello World")

#20210405 수업 #NUMPY
import numpy as np
import pandas as pd
import matplotlib as plt

#vertical 은 수직

a=np.random.randint(-3,3,10).reshape(2,-1) # -3부터 3까지 10개의 임의의 정수 추출 2행~열
b=np.random.randint(0,5,15).reshape(5,3)    # 0부터 5까지 15개의 임의의 정수 추출 5행 3열
print(a.shape,b.shape)

ab=np.matmul(a,b)
print(ab.shape,'\n') #\n -> enter를 하세요!
#(파이썬한테 지시)
# print(ab.shape,'\t') #\t -> tab을 눌러서
# 작동하세요.

np.matmul(a,b) #행렬의 곱
b= np.arange(16).reshape(4,-1)
print(b)

print(np.trace(b)) #대각선의 합

c=np.arange(25)
print(c)

d=np.array([[1,2],[3,4]])
print(np.linalg.inv(d))
print(d)
# non singular -> 역행렬 존재
# singular -> 역행렬이 존재 하지 않음.

#20210405 수업 #PANDAS
# missing data 처리가 용이
# Series 보다 DataFrame이 더 중요함.

frame=pd.DataFrame(np.arange(24).reshape(4,-1), 
columns =['c1','c2','c3','c4','c5','c6'], index =['r1','r2','r3','r4',])
# 0~23까지 4행~열로 행렬(numpy)을 만들건데, 그걸 데이터프레임(pandas)으로 만들거야
# 근데 컬럼(columns)을 'c1~'c6'으로 지정할거고 행(index)을 'r1'~'r4'으로 지정할거야.

print(frame)
print(frame[['c1','c2']])

print(frame.loc['r1':'r2'],['c2','c4','c3']) #frame.loc 할 때 원소 순서정렬

print(frame.loc['r1':'r2'],['c2','c3','c4'])
#print frame loc['r1:'r2],['c2']

#loc, iloc 행과 열로 접근한다. 
# iloc -> integer(우선순위가 행)
# lic -> label(우선순위가 열)

#DataFrame끼리 합할 때 합이 불가능한 원소는 NaN으로 구성된다.
# fill_value시에도 DataFrame에 둘 다 없으면 NaN으로 구성된다.

df=pd.read_csv('C:/Users/sundooedu/Desktop/data/kbo.csv')
print(df)
print(df.head()) #위에서 부터 디폴트 5개
print(df.tail()) #아래에서 부터 디폴트 5개

print(df.describe()) #count도 해주고, mean, var, std, qua 등등 기술통계량 값들을 보여줌
print(df.min(), df.max()) #axis 지정 가능
#series -> unique(): 중복 제거 유일한 값 남김.
#유일한 값 남김
#dataframe -> duplicated, drop_duplicates 
# map: list의 zip기능과 유사

# sum 함수에서 axis =0 이 행, axis=1 이 열
# 그러나 DataFrame에서는 axis=1이 행, axis=0이 열

frame = pd.DataFrame(np.arange(16).reshape(4,-1), 
                     columns = ['c1','c2','c3','c4'],
                     index = ['r1','r2','r3','r4'])
# print(frame.drop('r1')) #기본이 axis = 0
# print(frame.drop('c1', axis = 1))
# print(frame.drop(columns = ['c3','c4']))
# drop함수는 return이 없음. (적용 후 저장이 안 됨)
# drop 함수 후에 적용하고 싶으면 options
# option -> inplace
print(frame.drop(['r2'],inplace= True))  
#inplace= True로 고정을 시켜줘야 return(적용후 저장)이 됨.

print(frame)

obj = pd.Series(['apple','mango',np.nan,None,'Peach'])
obj
# print(obj.isnull().sum()) #결측치 개수 합
# print(obj.isna().sum) #결측치 없는 곳엔 False, 결측치 있는 곳엔 True

print(obj.dropna())  #결측치 제거 #리턴 안 됨
print(obj.dropna(thresh=1))
# thresh, any 두 개 이상일 때만 사용가능
# Dataframe에서 작동.

frame=pd.DataFrame([[np.nan,np.nan,np.nan,np.nan,np.nan],
                     [10,5,40,6,np.nan],[5,2,30,8,np.nan],
                     [20,np.nan,20,6,np.nan],
                     [15,3,10,np.nan,np.nan]])

print(frame.fillna(0)) #nan값을 전부 0으로 채워라
print(frame.fillna(frame.mean()))

print(frame.drop_duplicates()) #row입장에서 [T/F] 

# 데이터 변형 get_dummies -> 컴퓨터는 숫자밖에 인지하지 못하기 때문에 one-hot encoding
# 을 해주어야 한다.

df=pd.read_csv('C:/Users/sundooedu/Desktop/data/kbo.csv')
print(df)
print(df['팀'].unique())
print(df.shape) #데이터프레임의 행렬은 몇 행 몇 열?
print(df.columns) # 데이터프레임의 칼럼(열)나열
print(df.describe()) #데이터프레임의 기술 통계량
print(df.info()) #데이터 형, columns
print(df.groupby('팀').count()) # 팀별로 숫자를 세서 알려주세요.
print(df.groupby(['연도','팀']).sum()) #연도별로 팀별로 숫자를 합쳐서 알려주세요.
print(df.groupby('팀')['승률'].max()) #팀별 승률
print(df.groupby(['연도','팀'])['승률','순위'].max()) #연도별로 팀별로 승률과 순위를 알려주세요.

grouped = df.groupby('팀') 
for name, group in grouped:  #팀별로 연도별로 전부 출력 해줘.
    print(name)
    print(group)
    print('-'*50)


import matplotlib as plt

# matplotlib은 매틀랩에서 옴

%matplotlib qt  # 코드안에서 그림을 그려줘
# %: magic commend

plt.plot([1,2,3,4,5,6])
plt.show()

#scikit-learn으로 배우는 머신러닝

# 학습 데이터 세트와 테스트 데이터 세트는 서로 셔플되면 안 됨.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_iris
iris_dataset = load_iris()
print(iris_dataset)

#print(iris_dataset)
#target_names의 값은 우리가 예측하려는 붓꽃 품종의 이름을 문자열로 가지고 있다.
print('타깃의 이름: {}'.format(iris_dataset['target_names'], test_size=0.2, random_state=0))
# random_state=0 결과가 바뀌지 않게 고정

# scikit-learn은 항상 데이터가 2차원 배열일 것으로 예상. (3차원 이상 절대 안 나옴)

#20210406 수업
# 강의 커리큘럼
# 1. sql 
# 2. python 기초(python, numpy:선형대수, pandas, matplotlib)
# 머신러닝
# 3. Machine Learning Agressive 
# 3-1. Liner Regression
# 3-2. multiple regression
# 3-3. poisson regression(포아송분포)
# 4. Classification 
# 4-1. Decision tree
# 4-2. Boosting, Bagging, Random Forest 등등
# 5. Clustering
# 5-1. K-means, K-meroid 등등
# 5-2. Gaussian Mixed Model
# 5-3. DBSCAN
# 6 Bayesian -> Likelihood
# 6-1. LDA D Analysis
# 7. Support Vector Machine -> Kernel
# 딥러닝
# 8. perceptron(퍼셉트론)
# 9. muti perceptron
# 10. convolational neural vector
# 11. Recurrant Neural Vector

# 배우는데 필요한 수학
# 선형대수, 확률론, 수리통계학, 미적분학, 해석학

# 법선벡터 벡터1과 벡터2가 90도를 이룰 때, 기울기 곱은 -1이다.

# 머신러닝 회귀 예측의 핵심은 주어진 피쳐와 결정 값 데이터 기반에서 학습을 통해
# 최적의 회귀 계수를 찾아내는 것
# 직신이면 선형회귀 직선이 아니면 전부 비선형회귀

# 수치형 데이터 -> Regression(회귀)
# 범주형(카테고리) 데이터 -> Classification(분류)

# 회귀분석
# B(베타) = [B1,....,Bn]
# X = [X1,...,Xn]
# lim f(x) 좌극한과 우극한의 값이 다르면 미분 불가

# 비용 최소화 하기 - 경사 하강법 (Gradient Descent) 중요!!!!

# +,- 기호를 기울기에 따라 붙여줌 + 증가 - 감소

# 편미분이 시그마 안에 들어가려면 유한이여야 한다.

# 1차 편미분에서는 편미분 할게 2개지만 2차 편미분으로 넘어가면 편미분 할게 2^2=4가 됨.

# w1 과 w0 를 업데이트 할 w1_update, w0_update를 반환. 
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y)
    # 먼저 w1_update, w0_update를 각각 w1, w0의 shape와 동일한 크기를 가진 0 값으로 초기화
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    # 예측 배열 계산하고 예측과 실제 값의 차이 계산
    y_pred = np.dot(X, w1.T) + w0
    diff = y-y_pred
         
    # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성 
    w0_factors = np.ones((N,1))

    # w1과 w0을 업데이트할 w1_update와 w0_update 계산
    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))    
    
    return w1_update, w0_update

print(get_weight_updates)

import dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import seaborn as sns
##################
from sklearn import datasets #sklearn -> sklearn에 있는 패키지에 있는 데이터셋을 불러온 것
from sklearn.linear_model import LinearRegression
# linear_model 패키지에서 Linear Regression
from sklearn.model_selection import train_test_split, cross_val_score
# cross_val_score: K-fold cross validation.
from sklearn.metrics import mean_squared_error
# metric -> Squared error -> mean_squared error

boston= datasets.load_boston()

bos = pd.DataFrame(boston.data,columns = boston.feature_names)
# data -> boston.data/ columns -> boston.feature_names
# columns을 다 가지고 와! DataFrame -> numpy, pandas
# row, colums 표 형태인 데이터로 바꾸는 것.
# bos.head,bos.tail
print(bos.isnull().sum()) #결측치가 있으면 sum해주세요.
# =print(bos.isna().sum())
print(bos.describe())
bos['PRICE']=boston.target

sns.set(rc={'figure.figure':(11.7,8.27)})
plt.hist(bos['PRICE'], bins=30)
plt.xlabel('House Prices in $1000')
plt.show()

bos_1=pd.DataFrame(boston.data, columns=boston.feature_names)
corrlation_matrix=bos_1.corr().round(2) #R^2에 score를 계산해줘
# round(2) -> 소수점 두 번째 자리까지만 보여줘
sns.heatmap(data= corrlation_matrix, annot=True)
plt.show()

plt.figure(figsize=(20,5))

features = ['LSTAT','RM']
target = bos['PRICE']

for i,col in enumerate(features):        
    plt.subplot(1,len(features),i+1)
    x= bos[col]
    y= target
    plt.scatter(x,y,marker='o')
    plt.title('Variation in House Prices')
    plt.xlabel(col)
    plt.ylabel('House prices in $1000')
plt.show()

bos['PRICE']=boston.RM
X_rooms=bos.RM 
y_price=bos.PRICE

X_rooms=np.array(X_rooms).reshape(-1,2) 
y_price=np.array(y_price).reshape(-1,2)

print(X_rooms.shape)
print(y_price.shape)

#######
# Train/Test 분리
X_train_1,X_test_1,Y_train_1,Y_test_1=train_test_split(X_rooms, y_price, test_size=0.2, random_state=5)
print(X_train_1.shape)
print(X_test_1)
print(Y_train_1.shape)
print(Y_test_1.shape)

##### 
# 모델 적용
reg_1 = LinearRegression()
reg_1.fit(X_train_1, Y_train_1) # data하고 label을 같이 학습을 시킴

reg_1.score(X_train_1,Y_train_1)
print(reg_1.score(X_train_1,Y_train_1))
y_train_predict_1 = reg_1.predict(X_train_1)
rmse=(np.sqrt(mean_squared_error(Y_train_1,y_train_predict_1)))
r2= round(reg_1.score(X_train_1, Y_train_1),2) linear regressions score -> R^2

print("The model erformance for training set")
print("----------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

######## train은 끝 ######
# prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1) 
# plt.scatter(X_rooms,y_price)
# plt.plot(prediction_space, reg_1.predict(prediction_space), color = 'black', linewidth = 3)
# plt.ylabel('value of house/1000($)')
# plt.xlabel('number of rooms')
# plt.show()

X= bos.drop('PRICE',axix=1)
y= bos['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

reg_all=LinearRegression
reg_all.fit(X_train,y_train)
######### Train에 대한 평가 ####
y_train_predict = reg_all.predict(X_train)
rmse=(np.sqrt(mean_sqaured_error(y_train, y_train_predict)))
r2=round(reg_all.score(X_train, y_train),2)

print("The model erformance for training set")
print("----------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

import statsmodels as sm
X=bos.drop('PRICE', axis=1)
y=bos['PRICE']
X_constant = sm.add_constant(X)

### 요약 통계량 OLS 방법 -> R^2을 이용해도 됨. ###
model_1=sm.OLS(y,X_constant)
lin_reg=model_1.fit()

lin_reg.summary()














































