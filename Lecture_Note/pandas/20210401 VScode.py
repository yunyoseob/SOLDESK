print('Hello world')

import numpy as np
import pandas as pd

arr = np.array([1,2,3,4,5])
print(arr)
print(arr)
arr=np.random.random(29)
arr
arr.shape
arr.sum(axis=0).shape
arr.mean
import numpy as np
import pandas as pd
obj = pd.Series([0,1,2,3,4,5,6,7], index=['a','b','c','d','e','f','g','h'], dtype=np.int64)
obj
obj2=pd.Series(['a','b','c'], index=['i1','i2','i3'])
obj2
obj[[1,3,5]]
obj[obj<3] #array 이기 때문에 가능
obj['c']
obj['b']
obj[['e','c']]
obj['a':'c']
#obj[1:3] -> inter-location based
obj
obj['a':'c'] #label.index-location based
obj['d:e']=100 #d값과 e값을 100으로 수정
obj
obj.iloc[2] #inter location based property
obj.iloc[1:4] # label-based based propery
# 1. inter #2. label #3.iloc(1==3) 4. #loc(2==4)
frame=pd.DataFrame(np.arange(24).reshape(4,-1), columns=['c1','c2','c3','c4','c5','c6'], index=['r1','r2','r3','r4'])
frame
# iloc vs loc 차이 
obj.iloc[1:3] #iloc에서는 밸류의 위치를 넣고
obj.loc['a':'c'] #loc에서는 값 자체를 넣는다. #iloc가 loc보다 빠름
frame
frame.c3 #method, attribute
frame['c3'] # =frame.c3
frame[['c1','c2']] 
frame['r1':'r2']
frame['c1':'c2'] #칼럼을 기준으로 슬라이싱을 하게되면 아무런 값도 가져오지 않음
frame.iloc[[0],[3]] #r을 잡을 때는 특정 위치의 값을 잡아서 출력
frame.iloc[[0,1],1:4] #0,1행 1:4열가지고 오세요
frame.loc[['r1'],['c3']] #이런식으로 특정 위치의 값을 잡아도 괜찮음
frame.loc[['r1','r3'],:] #r1, r3 행 전체를 출력하고 싶으면 ,: 를 써서 출력해야함
frame.iloc[[0],[3]] #iloc -> interger 기반
frame.loc['r1':'r3',['c2','c3','c4']] #loc->label 기반
#add # 더하기 #radd #거꾸로 더하기
s1=pd.Series([1,2,3,4], index=['a','b','c','d'])
s2=pd.Series([10,20,30], index=['a','b','c'])
s1-s2
s1//s2 #not a number(NaN) = NOne
# DataFrame은 집합과 집합 연산시 둘 다 없는 경우는 NaN으로 채워짐
# DataFrane+Series 덧셈 가능. 단, index가 동일해야만 가능. 잘못하면 row로 할당되기 떄문.


# 2021년 4월 1일 수업
def add_all2(*args): #*args ->(): 수정이 가능.
    #list,tuple 다 허용이 됨.
    s=0
    for i in args:
        for j in i:
            s +=j
        return s

 # github 4/1 파이썬 자료에 자세한 설명있음. 꼭 볼 것
# 만약 pandas, numpy 안 깔려있으면 터미널 창에서 pip install pandas, pip install numpy
# upgrade 할거면 python -m pip install --upgrade pip
 
import matplotlib as mp


# df=pd.read_csv('C:/Users/sundooedu/Desktop/data/kbo.csv')
# print(df.describe)

#명백하게 확실하게 지워도 되는 데이터만 drop을 통해 데이터를 삭제함


# print(df1.drop('c2',axis = 1)) #column으로 접근해라.
# df1.drop([c3,c4] axis=1) axis 지정해야되고, return 
# print(df1.drop['r2'],inplace = True) #return -> 원래 것에 반영하라.

data1=pd.DataFrame({'id':['01','02','03','04','05','06'], 
                    'col':np.random.randint(0,50,6), 
                    'col2':np.random.randint(1000,2000,6)})


data2=pd.DataFrame({'id':['04','05','06','07'], 
                    'col':np.random.randint(0,50,4), 
                    'col2': np.random.randint(1000,2000,4)})                

print(data1, data2)
print(pd.merge(data1,data2,on='id')) # on option은 기준점을 어떻게 할거냐?
print(pd.merge(data1,data2,how='inner', on='id')) #on, how는 변수 id를 키로 잡고 교집합
print(pd.merge(data1,data2,how='outer', on='id')) #id를 키로 잡고 헙집합
print(pd.merge(data1,data2,how='left', on='id'))  #id를 키로 잡고 차집합
print(pd.merge(data1,data2,how='right', on='id')) #id를 키로 잡고 차집합

print(df.isnull().sum())
print(df.isna().sum().sum())

frame=pd.DataFrame([[np.nan,np.nan,np.nan], [10,5,40,6],[5,2,30,8],[20,np.nan,20,6],
                    [15,3,10,np.nan]], columns=['x1','x2','x3','x4'])

print(frame.fillna(15))

# frame=pd.DateFrame({'id':['0001','0002','0003','0001'],'name':['a','b','c']})

# print(frame.duplicated()) #중복 체크
# print(frame.drop_duplicates()) #중복 체크 겸 바로 drop
# frame.drop_duplicates(subset=['id'], keep = 'last')
# subset: 중복을 어떻게 할 것이냐?
# keep: 어떤 것을 남길지?

obj = pd.Series([10,-999,4,5,7,'n'])

print(obj.replace(-999,np.nan))
# list에서 썻던 것과 동일함.

print(obj.replace([-999,'n'],np.nan)) #브로드캐스팅

#연속형 데이터를 구간으로 나누어 범주화하는 방법

age =[20,35,67,39,59,44,56,77,28,20,22,80,32,46,52,19,33,5,15,50,29,21,33,48,85,80,30,10]
bins =[0,20,40,60,100]

cuts=pd.cut(age,bins) #연속형 변수를 이산형(범주형) 변수로 바꿔주는 것
print(cuts.codes)  

cuts=pd.cut(age,4, precision=1).value_counts()
print(cuts)

# get_dummies: categorical variable(명목형 변수)를 one-hot encoding 엔코딩(변환) 해줌

# df=pd.DataFrame({'col1':[10,20,30,40],
                'col2':['a','b','c','d']})
# print(df.get_dummies)

# df=pd.DataFrame({'col1':['001','002','003','004','005','006'],
                  'col2':[10,20,30,40,50,60]
                  'col3':['서울시','경기도','서울시','제주도','경기도','서울시']})

df=pd.read_csv('C:/Users/sundooedu/Desktop/data/kbo.csv')
# print(df['팀']).unique())
print(df)
print(df.info()) #30개의 행 9개의 열로 구성
print(df.groupby('연도','팀')





















































































