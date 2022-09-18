print('Hello world')

import numpy as np
import pandas as pd

array = np.array([1,2,3,4,5])
arrprint(arr)
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






























