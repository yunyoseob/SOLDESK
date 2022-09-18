print("Hello World")

#20210405 수업 #NUMPY
import numpy as np
import pandas as pd

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


































































