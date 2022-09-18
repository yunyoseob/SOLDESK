#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np


# In[72]:


import pandas as pd


# In[73]:


import matplotlib as plt


# In[74]:


# w1 과 w0 를 업데이트 할 w1_update, w0_update를 반환. 
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y) #y= w_0 + w_1 X_1 -> 벡터의 길이
    # 먼저 w1_update, w0_update를 각각 w1, w0의 shape와 동일한 크기를 가진 0 값으로 초기화
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    # 예측 배열 계산하고 예측과 실제 값의 차이 계산
    y_pred = np.dot(X, w1.T) + w0 #np.matmul 써도 되지만, 
    #어차피 벡터 계산이기 때문에 dot을 썻음.
    diff = y-y_pred #error function = [실제값 - 예측값]
         
    # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성 
    w0_factors = np.ones((N,1)) #초기값 ones로 셋팅 N크기만큼 받아들이고,

    # w1과 w0을 업데이트할 w1_update와 w0_update 계산
    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff)) #error ftn: mse(mean sqaure error)
    # /summation_i^n(y-y)hat)(-x_i)
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))    
    #summation_i^n(y-y_hat)(-x_1)
    
    return w1_update, w0_update #w_0,W_1 update


# In[75]:



# 입력 인자 iters로 주어진 횟수만큼 반복적으로 w1과 w0를 업데이트 적용함. 
def gradient_descent_steps(X, y, iters=10000):
    # w0와 w1을 모두 0으로 초기화. 
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    
    # 인자로 주어진 iters 만큼 반복적으로 get_weight_updates() 
    #호출하여 w1, w0 업데이트 수행. 
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update #w1(왼쪽에 있는) -> new, w_1(오른쪽에 있는) -> old
        # w1_update= gradient descent 방법
        # new = old - update (update = 0 -> new = old) 최적의 값을 찾음.
        w0 = w0 - w0_update
              
    return w1, w0


# In[76]:


def get_cost(y, y_pred):
    N = len(y) 
    cost = np.sum(np.square(y - y_pred))/N
    # root(실제값-예측값) 다 더해서 저장한게 cost
    return cost

w1, w0 = gradient_descent_steps(X, y, iters=1000) #1000번을 반복
#최적의 값을 뽑고 그 때의 cost값을 출력.
print("w1:{0:.3f} w0:{1:.3f}".format(w1[0,0], w0[0,0]))
y_pred = w1[0,0] * X + w0
print('Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))


# In[77]:


plt.scatter(X, y)
plt.plot(X,y_pred)


# In[78]:


# data가 적으면 gradient descent 방법을 사용할텐데, data가 굉장히 큼.
# -> 미분자체가 계산량이 많아짐/변수가 많아서 미분이 많아짐.
# 통계에서는 모집단(전체) -> 표본 통계량 혹은 결론 stochastic


# In[79]:



def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=1000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 100000
    iter_index =0
    
    for ind in range(iters):
        np.random.seed(ind)
        # 전체 X, y 데이터에서 랜덤하게 batch_size만큼 데이터 추출하여 
        #sample_X, sample_y로 저장 
        #(참고 : https://medium.com/@shistory02/numpy-permutation-vs-shuffle-34fe56f0c246)
        # Stochastic gradient descent / Mini-batch graident descnet (참고 : https://nonmeyet.tistory.com/entry/Batch-MiniBatch-Stochastic-%EC%A0%95%EC%9D%98%EC%99%80-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EC%98%88%EC%8B%9C)
        stochastic_random_index = np.random.permutation(X.shape[0])
        #permutation: array를 복사해서 셔플을 한다.
        #shuffle은 array를 셔플해서 INPLACE를 한다. : array 자체가 변함.
        sample_X = X[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]
        # 랜덤하게 batch_size만큼 추출된 데이터 기반으로 
        # w1_update, w0_update 계산 후 업데이트
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    
    return w1, w0


# In[80]:


w1, w0 = stochastic_gradient_descent_steps(X, y, iters=1000)
print("w1:",round(w1[0,0],3),"w0:",round(w0[0,0],3))
y_pred = w1[0,0] * X + w0
print('Stochastic Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))


# In[81]:


import numpy as np
import matplotlib.pyplot as plt

lr_list = [0.001, 0.5, 0.3, 0.77]

def get_derivative(lr_list):

  w_old = 2
  derivative = [w_old]

  y = [w_old ** 2] # 손실 함수를 y = x^2로 정의함.

  for i in range(1,10):
    #먼저 해당 위치에서 미분값을 구함

    dev_value = w_old **2

    #위의 값을 이용하여 가중치를 업데이트
    w_new = w_old - lr * dev_value
    w_old = w_new

    derivative.append(w_old) #업데이트 된 가중치를 저장 함,.
    y.append(w_old ** 2) #업데이트 된 가중치의 손실값을 저장 함.

  return derivative, y

x = np.linspace(-2,2,50) 
x_square = [i**2 for i in x]

fig = plt.figure(figsize=(12, 7))

for i,lr in enumerate(lr_list):
  derivative, y =get_derivative(lr)
  ax = fig.add_subplot(2, 2, i+1)
  ax.scatter(derivative, y, color = 'red')
  ax.plot(x, x_square)
  ax.title.set_text('lr = '+str(lr))

plt.show()


# In[115]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import dataset
import datasets


# In[83]:


import pandas as pd   
test=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/test.csv')  #캐글 타이타닉문제
test

train=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/train.csv')
train

gender_submission=pd.read_csv('C:/Users/sundooedu/Desktop/titanic/gender_submission.csv')
gender_submission


# In[124]:


def bar_chart(feature):
     survived=train[train['Survived']==1][feature].value_counts() #생존자를 카운트
     dead=train[train['Survived']==0][feature].value_counts() #사망자를 카운트
     df=pd.DataFrame([survived,dead]) #[생존자, 사망자]를 dataFrame
     df.index=['Survived','Dead'] #index화
     df.plot(kind='bar',stacked=True, figsize=(10,5)) #그림을 그림


# In[125]:


train_test_data = [train,test]
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[126]:


train['Title'].value_counts()


# In[127]:


# one-hot encoding
title_mapping= {'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':3, 'Rev':3,
               'Col':3, 'Major':3, 'Mme':3, 'Don':1, 'Ms':1, 'Lady':1,'Sir':1, 
               'Countess':3, 'Capt':3, 'Jonkheer':3 }
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)


# In[128]:


bar_chart('Title') 


# In[129]:


train.head()


# In[130]:


sex_mapping={'male':0,'female':1}
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map(sex_mapping) 


# In[157]:


#missing Age를 각 Title에 대한 연령의 중간값으로 채움(Mr,Mrs,Miss,others)
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace = True)

test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace = True)


# In[158]:


facet=sns.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sns.kdeplot,'Age',shade=True) #kde: 이차원 밀집도 그래프
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
sns.axes_style('dark')

plt.show()


# In[159]:


facet=sns.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sns.kdeplot,'Age',shade=True) #kde: 이차원 밀집도 그래프
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
sns.axes_style('dark')

plt.xlim(0.20)
plt.show()


# In[160]:


# for dataset in train_test_data: # train.test 도이시에 접근
## loc 한꺼번에 잡았는데 불가함 Multi index

for dataset in train_test_data:
    dataset.loc[dataset['Age'] <=16, 'Age_bin']=0
    dataset.loc[(dataset['Age']>16)& (dataset['Age']<=26), 'Age_bin']=1
    dataset.loc[(dataset['Age']>26)& (dataset['Age']<=36), 'Age_bin']=2
    dataset.loc[(dataset['Age']>36)& (dataset['Age']<=46), 'Age_bin']=3
    dataset.loc[(dataset['Age']>26), 'Age_bin']=4    

print(train['Age'])


# In[161]:


train2= train.copy()
train2.loc[(train2.Age<=16),"Age_bin"]=0
train2.loc[(train2.Age>16) & (train2.Age<=26),"Age_bin"]=1
train2.loc[(train2.Age>26) & (train2.Age<=36),"Age_bin"]=2
train2.loc[(train2.Age>36) & (train2.Age<=46),"Age_bin"]=3
train2.loc[(train2.Age>46),"Age_bin"]=4
train2


# In[162]:


Test2= train.copy()
Test2.loc[(train2.Age<=16),"Age_bin"]=0
Test2.loc[(train2.Age>16) & (train2.Age<=26),"Age_bin"]=1
Test2.loc[(train2.Age>26) & (train2.Age<=36),"Age_bin"]=2
Test2.loc[(train2.Age>36) & (train2.Age<=46),"Age_bin"]=3
Test2.loc[(train2.Age>46),"Age_bin"]=4
Test2


# In[175]:


train2.isna().sum()


# In[174]:


train2=train2.fillna(0)


# In[164]:


# FamilySize 함께 동승한 부모님과 아이들의 수화 형제와 배우자의 수
# 혼자탄거랑 가족들이랑 탄거랑 어떻게 다른지?
# Sibsp+parch


# In[177]:


train2['FamilySize']=train2['SibSp']+train2['Parch']+1
Test2['FamliySize']=Test2['SibSp']+Test2['Parch']+1


# In[178]:


facet = sns.FacetGrid(train2, hue ='Survived', aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade = True) # kde : 이차원 밀집도 그래프
facet.set(xlim=(0,train2['FamilySize'].max()))
facet.add_legend()

plt.show()


# In[180]:


# 가설: 혼자일 경우는 사망률이 높다.
X_train = train.drop(['Survived','PassengerId'],axis=1)
Y_train = train['Survived']
X_test = test.drop('PassengerId',axis = 1).copy()


# In[181]:


X_train


# In[182]:


X_test['Fare'].fillna(0,inplace=True)
X_test['Title'].fillna(0,inplace=True)


# In[183]:


X_test.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




