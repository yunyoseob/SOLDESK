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

#2021년 4월 2일 금요일
#파이썬~판다스 총 복습
# 변수 = 연산/조건/함수
# [1:] -> 2번째 부터 끝까지 indexing
# [:] -> 전체 indexing
# [::-1] 전체를 뒤집음 가나다라 -> 라다나가
# replace 대체 , split 자르기 중요
# join함수는 잘 안씀
# list 중요
list_1=[3,2,5]
list_1
type(list_1)
# list 인덱싱은 가장 바깥쪽 괄호부터 시작
# list 삽입,수정,삭제에서 삽입시 append,insert를 많이 씀. append는 리스트 제일 뒤에
# append는 리스트 제일 뒤에, insert는 원하는 위치에 쓴다.
# list 수정시 인덱스나 슬라이싱을 통해 수정
# list 삭제시 remove를 쓴다.
#ㅣist 꺼내올시 pop을 쓴다. 그치만 remove를 훨씬 많이 씀.
# list 정렬시 sort를 씀. reverse=True 내림차순, reverse=False 오름차순

# 자료구조: stack -> 후입선출 (Last in First Out)

# 패킹: 한 변수에 여러 개의 데이터를 넣는 것
# 언패킹: 한 변수의 데이터를 각각의 변수로 반환
e=[1,2,3] #1,2,3이라는 변수를 t에 패킹
e  

f,g,h=e     #e에 있는 값 1,2,3을 변수 a,b,c에 언패킹
print(f,g,h)
# 1 2 3 으로 출력됨.

# 딕셔너리 key를 통해 value값에 접근 가능 무조건 key로 접근
--

# SET(집합)
# {}을 이용하여 set선언
set_1= set()
set_1
set_4={3,5,'hi'}
set_4

# 합집합과 교집합은 union과 intersection이 있지만 그냥 &와 |를 쓰면 편함
set_5={1,2,3,4}
set_6={3,4,5,6}
set_5&set_6 #교집합
set_5 | set_6 #합집합
set_5 - set_6 #차집합
set_6 - set_5

# 조건문
password = 111111

if password == 123456:
    print('비밀번호가 풀렸습니다.')
else:
    print('비밀번호가 틀렸습니다.')

# 조건문 안에 조건문
money=20000
card=True

if card: #카드가 참이라면
    if money <20000: #근데 머니가 20000미만이라면
        print("삽겹살을 먹는다")
    elif money >20000:
        print("장어를 먹는다")
    else: 
        print("고스톱을 친다")
        
# vscode에서는 int(input()) 출력이 안 됨.
# (while,for pass) 무시하고 지나가라
# (while,for) continue 다시 while로 가라
# (while, for) break 탈출

#연습문제3 파이썬 p.173풀이
x=[3,6,9,20,-7,5]

#3-1번
for i in range(len(x)):
    x[i]=x[i]*10
print(x) #요소 하나씩 접근을 하여서 곱을 해주세요! # 결과는 리스트로

#3-2번
y={'math': 70,  'science':80, 'english':20}
#dict->key로 접근을 해야만 value가 따라옴.
for key in y:
    val=y[key]
    val =y[key]+10
    print('%s &d' %(key,y[key])) #dict이 제대로 안쓰임
    # print('%s %d' %key,y[key)

#3번은 패스
#4번
word = ['school', 'game', 'piano','science','hotel','mountain']

new_word=[]
for i in range(len(word):
    if len(word[i]) >= 6:
        new_word.append(word[i])

#6번
c=0
d=1

while (d==1):
    a=int(input())
    if (a=='s') or (a=='S'):
        d=0
    else:
        a=int(a)
        c+=a


# 7번
kor_score=[39,69,20,100,80]
math_score=[32,59,85,30,90]
eng_score=[49,70,48,60,100]

midterm_score=[kor_score, math_score, eng_score]
#2차원리스트에 접근합니다. 긴장하세요.
# list(zip(kor_score, math_score, eng_score))으로 접근해도 됨.

student_score[0,0,0,0,0]
i=0
#빈칸 생성
for subject in midterm_score:  #kor? math? eng?
#kor[0]+math[0]+eng[0]
#row(행)
    for score in subject: #과목을 받았으니까
        #한번 더 접근하면 각각 score
        student_score[i] += score
    i=0
else:
    a,b,c,d,e = student_score
    student_average = [a/3,b/3,c/3,d/3,e/3]

#가위바위보 문제
import random

while True:
    user = input('가위바위보를 하세요.')
    is user == '가위':
        if random.choice(['가위','바위','보']) == '가위':
            print('무승부')
        elif random.choice(['가위','바위','보'])=='바위':
            print('승')
        else:
            print('패')
    is user == '바위':
        if random.choice(['가위','바위','보']) == '가위':
            print('패')
        elif random.choice(['가위','바위','보'])=='바위':
            print('무승부')
        else:
            print('승')
    is user == '보':
        if random.choice(['가위','바위','보']) == '가위':
            print('승')
        elif random.choice(['가위','바위','보'])=='바위':
            print('패')
        else:
            print('무승부')        

#p.209 연습문제 
def func(string, unit=2):
    i=0
    while i < len(string):
        print(string[i:i+unit])
        i+=unit

func('테스트를 위한 문장입니다.',2)

#p.210
# 첫 번째 방법
def add_all(*inputs):
    s=0
    for s in range(len(input)):
        s+=input[i]
        return s
# 두 번째 방법
def add_all2(*inputs): #list는 받아들임 #tuple은 안 됨.
    s=0
    for i in input:
        for j in i:
            s+=j
        return s
def add_all3(*inputs)

people=['펭수','뽀로로','뚝딱이','텔레토비']

def func1_with_enu(line):             #전역변수
    new_lines=[]
    for idx, val in enumerate(line):
        print("대기번호 %d번: %s" %(idx+1,val))
        new_lines.append((idx+1, val))
    return new_lines

lines=func1_with_enu(people)   #지역변수

# zip을 tuple으로 묶는 이유: 수정이 힘들어서
# lambda: 파이썬 기본과정으로 쓰이기 보다는 데이터 분석과정에서 쓰이기 떄문에 배움

sentence=['코로나', '바이러스를', '예방하기', '위해','사회적 거리두기를','실천합시다.','마스크를', '끼고', '손씻기를', '생활합시다.']
for i in range(len(sentence)):
    sentence[i]=str(i)
str_sentence=list(map(lambda x:str(x),sentence))

#밖으로 꺼내고 싶으면 classmethod

#NUMPY 
#list로 잡기 너무 크기 때문에 행렬로 넘김

    
#np.array [1 2], [3 4] -> ,는 다음 행으로 넘기라는 의미
# 이미지는 3차원, 음성은 1차원

#full함수는 넘겨도 됨
#_like도 넘겨도 됨
#Hstack 행으로 결합
#Vstack 열로 결합

#로또번호 생성기 만들기
import numpy as np

def make_lotto(count):
    for i in range(count):
        lotto_num=[]
        for j in range(6): #6번 반복 굳이 없어도 됨.
        lotto_num=numpy.random.choice(range(1,46)),6,replace
        lotto_num.sort()
    print('{}. 로또번호:{}'.format(i+1,lotto_num))

# p.55 연습문제 2)



























