#!/usr/bin/env python
# coding: utf-8

# In[1]:


2**2**2


# In[2]:


10**2


# In[6]:


10*10


# In[8]:


a=3
b=5
print(a+b)


# In[9]:


a=3


# In[10]:


b=5


# In[11]:


print('a+b') #a+b를 문자로 인식하세요. '' -> string(문자열)


# In[12]:


a=a+1 # a에서 하나 증감 /commnet: 값을 주석처리(실행불가)
a+=1 # a=a+1 #a를 1을 더해서 update(값이 변한 것)를 해줘라


# In[14]:


apple = 5 #1번 문제


# In[15]:


orange = 3


# In[16]:


total = apple + orange


# In[17]:


print(total)


# In[22]:


kor = 100 #2번 문제


# In[23]:


eng = 88


# In[24]:


math = 94


# In[25]:


avg = (kor+eng+math)/3


# In[29]:


print(avg)


# In[30]:


a=1.0 # integer -> .이라는 소수점이 있으면 무조건 float
print(type(a))


# In[37]:


a= 5.3 #a와 b를 정수로 변환하고 나서 더하기
b=5.7

a=int(a)
b=int(b)

print(a+b)


# In[40]:


f='문자형 예시입니다.' #작은 따옴표 혹은 큰 따옴표를 하면 string(문자)으로 간주
print(f)
type(f)


# In[41]:


a = '3' #자료형은 무엇인가?
print(type(a))


# In[46]:


print('a\nb') # \n -> enter \t -> tab


# In[52]:


str_5='인덱싱을 하기 위한 string입니다.'
str_5[2:7]


# In[57]:


a= 'alphabet'
print(a.lower()) #이렇게만 선언하면 안되나요? print(a.lower)와 print(a.lower() )
#무슨 차이인가요? print(a.lower)로 치면 오류 나옴. upper도 마찬가지


# In[62]:


a='alphabet' #문자열에서 가장 앞에 있는 것을 출력함
a.find('a')


# In[63]:


list_1 = [1,2,3,4,5,6]


# In[64]:


type(list_1)


# In[65]:


len(list_1)


# In[66]:


list_1[1]


# In[67]:


list_1


# In[68]:


list_3 = [1,2,'문자',['이중리스트','가능'],('리스트 속','튜플')]


# In[70]:


list_3


# In[71]:


list_3[3]


# In[75]:


list_3[3][1]


# In[78]:


list_3[0]=100


# In[79]:


list_3


# In[80]:


tuple_1=(1,2,3,4,5)


# In[82]:


tuple_1


# In[37]:


food = ['삼겹살','스테이크','회','소고기','장어']
food


# In[38]:


food.insert(0,'파이썬')


# In[39]:


food.append('집') ## food.append(5,'집')도 가능하다.


# In[40]:


food.pop(0)


# In[41]:


food


# In[42]:


food.remove('집')


# In[43]:


food


# In[46]:


dict_3={ '홍길동':100, '홍계월':200, '슈퍼맨':500, '배트맨':250}
dict_3


# In[59]:


63 <= 65


# In[60]:


70 >= 65


# In[61]:


64 <= 65


# In[73]:


a = (63,70,64)
print(a)


# In[2]:


student_score=[0,0,0,0,0]
i=0
for subject in mid_score: #kor,math,eng: 과목 선택
    for score in subject:
    i=0
    for subject in mid_score:
        for score in subject:
            student_score[i] +=score 
            i +=1
            
            i=0
            else:
    student_score[i] += score
    print(subject,score,'i: ',i,student_score)
    i +=1
    
    i=0
else:
    a,b,c,d,e=student_score
    student_average=[a/3,b/3,c/3,d/3,e/3]
    print(student_average)


# In[3]:


students = {}


# In[15]:


with open('./data/students.txt'.'r'.encoding= 'UTF-8') as f:
    data=f.read()
    print(data)


# In[17]:


file1=open(r'C:\Users\sundooedu\Desktop\data','r',encoding='utf-8')
text=file1.read()
print(text)


# In[23]:


with open('C:/Users/sundooedu/Desktop/data/students.txt','r',encoding='UTF-8') as f:
    data=f.read()
    print(data)
#   print(data)
    lines=data.split('\n')
    for stu in lines:
        s=stu.split('\t')
        students[s[0]]=s[1:]
        print(students)
     


# In[24]:


students


# In[25]:


def sub(a,b):
    print('뺄셈의 결과는 %d입니다.' %(a-b))
    return


# In[26]:


d = sub(1,2)


# In[27]:


print(d)


# In[29]:


def f(x):
    print(x+10)


# In[31]:


f(10)


# In[32]:


c=f(10)


# In[33]:


print(c)


# In[36]:


list_ex=[5,4,3,2,1]

sorted(list_ex) #return이 있음
list_ex.sort #변화는 일어나고, 보여지지 않았음 -> return 함수가 없음


# In[37]:


def f(x):
    print(x+10) #print만 하는데


# In[38]:


def f(x):
    return x+10 #return -메모리까지 할당을 시키고 값도 할당


# In[40]:


for i in range(1,10):
    print(f'==={i} 단===')
    for j in range(1,10):
        print(i*j)


# In[56]:


def div(a,b):
    return(a//b,a%b)


# In[57]:


div(10, 3)


# In[58]:


div(3, 5)


# In[59]:


div(1,10)


# In[63]:


def div2(a,b):
    if a<b:
        big=b
        small=a
    else:
        big=a
        small=b
    if small ==0:
        print('0은 사용할 수 가 없습니다.')
    else:
        q=big//small
        r=big%small
        return(q,r)


# In[64]:


div2(3,5)


# In[65]:


div2(0,5)


# In[66]:


div2(10,1)


# In[67]:


def test(*args): #argumnet -> 공간만 만들음(args: tuple)
    print(args)


# In[68]:


def test(*args):
    print(args)


# In[69]:


test(1,2,3,4,5)


# In[70]:


test(1,2,3,4,5,6,7,8,9,10)


# In[71]:


def test1(**kwargs): #kwargs: dict를 받을 것입니다. 미리 공간 생성.
    print(kwargs)


# In[72]:


test1(a=1, b=2, c=3)


# In[84]:


#p.209 연습문제
def func(string,unit=2):
    i=0
    while i <len(string):
        print(string[i:i+unit])
        i += unit
    
    


# In[86]:


func('테스트를 위한 문장입니다.',4)


# In[94]:


def add_all(*inputs):
    s=0
    for i in range(len(inputs)):
        s +=inputs[i]
    return s


# In[95]:


add_all(1,2,3,4,5,6,7,8,9,10)


# In[100]:


def add_all(*args):
    print(args)


# In[101]:


add_all(1,2,3,4,5,6,7,8,9,10)


# In[105]:


# **awargs -> list만 받는다는 가정
def add_all2(*args):
    s=0
    for i in args:
        for j in i:
            s += j
        return s


# In[106]:


add_all2([1,2,3,4,5,6,7,8,9,10])


# In[122]:


people =['펭수','뽀로로','뚝딱이','텔레토비']

def func1(line):
    new_lines=[]
    for idx.val in enumerate(line):
        print('대기 번호 %d번 : %s' ^(idx+1, val))
        new_lines.append((idx+1,val))
        return new_lines
    


# In[110]:


st = 'abcd'

for x in enumerate(st):
    print(x)


# In[119]:


se = {0:'p',1:'b',2:'d'}


# In[120]:


type(se)


# In[121]:


for x in enumerate(se):
    print(x)


# In[124]:


people =['펭수','뽀로로','뚝딱이','텔레토비']

def func1(line):
    new_lines=[]
    i=1
    for x in line:
        print('대기 번호 %d번 : %s' %(i,x))
        new_lines.append((i,x))
        i+=1
    return new_lines
    


# In[125]:


func1(people)


# In[126]:


str_list = ['one','two','three','four']
num_list = [1,2,3,4]

for i in zip(str_list, num_list): #data가 두 개 이상으로 쪼개져있을 때,
    #data를 생성할 떄 zip 함수를 씀
print(i)


# In[127]:


def plus_two(num):
    return num+2


# In[128]:


a=2
b= plus_two(a)
print(b)


# In[129]:


func2=lambda x:x+2


# In[131]:


c=func2(2)


# In[132]:


c


# In[135]:


items = [1,2,3,4,5]

squared = []
for i in items:
    squared.append(i*i)
    print(squared)


# In[136]:


squared_map = list(map(lambda x:x**2, items)) #map 쓰지 않았을떄는
# fun = lambda x: x**2
# c = fun(2)
#map(lambda를 지정 -> item를 적용해주세요.)


# In[137]:


squared_map= list(map(lambda x:x**2, items))


# In[138]:


squared_map


# In[139]:


items = [1,2,3,4,5]

str_items = list(map(lambda x: str(x),items))
print(str_items)


# In[141]:


items = [1,2,3,4,5]

for i in range(len(items)):
    items[i]=str(i)
str_items = list(map(lambda x: str(x),items))
print(str_items)


# In[149]:


list1=[0,1,2,3,4,5,6,7,8,9]
list2=[]
for x in range(10):
    list2.append(x)
print(list2)


# In[150]:


lc_1=[x for x in range(10)] # for x in range(10) -> x
# [저장할 값 for 원소 in 반복가능객체]
#[x for x in range(10) if x%2 ==0]
print(lc_1)


# In[156]:


#파이썬심화 list comprehension
# 1) 구구단 출력
tables=[2*x for x in range(1,10)]


# In[163]:


sentence='코로나 바이러스를 예방하기 위해 사회적 거리두기를 실천합시다.외출 시에는 마스크를 끼고, 손씻기를 생활화 합니다.'


# In[164]:


sentence


# In[165]:


len_sen=[len(s) for s in sentence.split()]
print(len_sen)


# In[166]:


list_1=(1,4,9,16,25,36,49)


# In[167]:


print(list_1)


# In[168]:


lc_3=[x**2 for x in range(1,11) if x**2>50]


# In[169]:


lc_3


# In[170]:


sentence


# In[172]:


len_condition=[s for s in sentence.split() if len(s)<5]


# In[173]:


len_condition


# In[176]:


list_4=[]
for x in range(1,11):
    if x%2 ==0: #짝수면 작동
        list_4.append(x**2)
    else: #홀수면 작동
        list_4.append(x**3)


# In[175]:


list_4


# In[178]:


list_5=[]
    if x<40:
        print(x+5)
    else:
        print (41)


# In[179]:


list_5=[12,67,32,48,19,57,29,49]


# In[180]:


lc_6=[x+5 if x<=40 else 41 for x in list_5]
#lc_6[x+5 for x in line_5 if x<=40 else 41] -> 이렇게 하면 무조건 에러


# In[181]:


lc_6


# In[184]:


students = {'보라돌이':61, '뚜비':35, '나나':78, '뽀':88}
type(students)


# In[197]:


# students.items()

result = [(name,True) if score>=60 else (name,False) for name.score in students.items()]


# In[191]:


students


# In[208]:


class test():
    name='아무나'
    age=0
    def __init__(self, name, age): #name, age를 variable로 받아서
        print('생성자 호출!')       #인스턴스로 내려줌
        self.name=name       #name -> self.name으로 받고
        self.age=age         #age -> self.age로 받음 -> 그 다음 내려주
    def __del__(self):
        print('소멸자 호출!')
    def info(self):
        print('나의 이름은', self.name, '입니다!')
        print('나이는', self.age, '입니다!')


# In[209]:


r=test('류영표',31) #인스턴스를 안 받음


# In[210]:


r.info() 


# In[216]:


class Myphone():
    def __init__(self, model, color):
        self.model = model
        self.color = color
    def set_name(self, name):
        self.user = name
        print('사용자의 이름은: %s' %self.user)
    def set_number(self,number):
        self.number = number


# In[217]:


class Myphone2(Myphone):
    def has_case(self,val=False):
        self.case=val


# In[227]:


p2=Myphone2('갤럭시 노트10+','무지개색')


# In[228]:


p2.color


# In[229]:


p2.model


# In[230]:


class Myphone2(Myphone):
    def has_case(self,val=False):
        super().__init__


# In[231]:


p2.color


# In[239]:


class Human():
    def __init__(self, brith_date, sex, nation):
        self.birth_date = brith_date
        self.sex = sex
        self.nation = nation
    def give_name(self,name):
        self.name=name
        print('이름은 &s 입니다' %self.name)
    def can_sing(self, ability):
        if ability == 'True': 
            print('Sing a song')         


# In[240]:


YP=Human(199109,'M','한국')


# In[241]:


YP.birth_date


# In[242]:


YP.can_sing('True')


# In[243]:


YP.sex


# In[254]:


class ChildSuper(Human):
    def __init__(self,birth_date,sex,nation,eye):
        super().__init__(birth_date,sex,nation)
        self.eye=eye
    def can_sing(self):
        print("Cant's sing")
    def can_dance(self):
            print('Dance Time!')


# In[1]:


import numpy as np #as : 별명~


# In[18]:


A=[1,2,3,4]
a=np.array(A,str)


# In[19]:


print(a)


# In[22]:


arr = np.array([[1,2,3],[4,5,6]])
arr


# In[23]:


arr+3


# In[24]:


arr*3


# In[25]:


256*256


# In[28]:


a=np.array([1,2,3,4,5])


# In[29]:


print(a)


# In[30]:


print(type(a))


# In[31]:


arr.dtype


# In[35]:


float_arr=arr.astype(np.str)


# In[36]:


float_arr.dtype


# In[37]:


float_arr


# In[52]:


arr=np.array([[[[1,2],[4,5],[5,6]]]])


# In[53]:


print(arr)


# In[54]:


arr.size


# In[55]:


arr.shape


# In[56]:


arr.ndim


# In[57]:


print(arr.ndim)


# In[64]:


a=np.array([[[[1,2],[3,4],[5,6],[7,8]]]])


# In[65]:


print(a)


# In[66]:


print(a.shape)


# In[67]:


a=np.array(range(20))


# In[68]:


print(a)


# In[69]:


arr5=np.array([[10,20],[40,50]])


# In[70]:


arr5


# In[74]:


arr6=np.array(((1,2),(3,)))
print(arr6)


# In[75]:


import numpy as np


# In[76]:


a=np.array([1,2,3])
b=np.array([4,5,6])


# In[77]:


print(a,b)


# In[79]:


np.hstack([a,b])


# In[82]:


np.concatenate((a,b),axis=0)


# In[83]:


np.vstack([a,b])


# In[84]:


np.concatenate((a,b),axis=1)


# In[86]:


a=np.array([[13,16,20,22,27,43],[21,27,31,36,37,45],[4,15,16,26,30,39],[7,15,26,29,40,41]])
a


# In[91]:


import numpy as np
def make_lotto(count):
    for i in range(count):
        lotto_num = []
        for j in range(6):
            lotto_num=np.random.choice(range(1,46),6, replace=False)
            lotto_num.sort()
        print('{} 로또 번호 {}'. format(i+1, lotto_num))

count=int(input('로또 번호를 몇개 생성할까요?'))
make_lotto(count)


# In[93]:


import random
LottoNumber = []
while len(LottoNumber) <6:
    V=random.randint(1,45)
    if V not in LottoNumber:
        LottoNumber.append(V)
print(LottoNumber)


# In[99]:


import random
num= input('게임 수:')

for i in range(0,int(num)):
    lotto=random.sample(range(1,46),6)
    lotto.sort()
    print(lotto)


# In[98]:


import random

sh = int(input('True:1, False: 0'))
while(sh==1):
    lotto = int(input('로또를 몇장 사시겠습니까?'))
    if lotto <=0:
        print('종료합니다')
        break
    while lotto>0:
        print('랜덤하게 생성된 로또 번호입니다')
        for i in range(1,lotto+1):
            print('[%d]:' %i, end='')
            for j in range(6):
                print("{0:3d}".format(random.randint(1,45)), end='')
            print()
        break


# In[100]:


import numpy as np


# In[103]:


A=np.ones([3,3]) 


# In[104]:


print(A)


# In[105]:


A.flatten()


# In[108]:


arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12])


# In[109]:


arr.shape


# In[110]:


arr.size


# In[116]:


arr.reshape(3,4)


# In[148]:


arr=np.arange(20).reshape(4,5)


# In[135]:


arr


# In[138]:


arr.reshape(4,5)


# In[149]:


arr.transpose()


# In[159]:


print(arr.transpose().shape)


# In[160]:


a


# In[161]:


a.shape


# In[162]:


y=np.swapaxes(a,0,1) #0은 가장 높은 차수의 축, 1은 그 다음 높은 차수의 축 
# 즉, 원소의 행과 열을 바꾸라는 것 (reshape와 다르게 작동,)
y


# In[163]:


a=np.arange(6).reshape(1,2,3)


# In[164]:


a


# In[166]:


a.shape


# In[167]:


x=np.swapaxes(a,1,2) # 0 -> 가장 큰 차수(X) 1->2차원 축하고 2->1차원 축을 바꿔라.


# In[168]:


x


# In[169]:


y=np.swapaxes(a,0,1) #3차원 축과 2차원 축을 바꿔라


# In[170]:


y


# In[171]:


z=np.swapaxes(a,0,2) #3차원 축과 1차원 축을 바꿔라


# In[173]:


z


# In[ ]:


# (1x2x3) -> (2x1X3)


# In[176]:


arr=np.arange(30).reshape(3,2,-1)


# In[177]:


arr.shape


# In[178]:


arr


# In[179]:


arr.transpose().shape


# In[180]:


arr.transpose((1,0,2)).shape #0이 3차원에 있음 2는 2행 5는 열
#transpose axis 절대 안 건드립니다.


# In[183]:


arr.transpose() #axis를 건드릴 수 있지만 절대 안 함.
arr.T


# In[188]:


arr=np.arange(20)


# In[195]:


arr1=[]
arr2=[]
for i in arr:
    if arr[i]%2==0:
        arr1.append(arr[i])
    else:
        arr2.append(arr[i])


# In[196]:


arr1


# In[197]:


arr2


# In[200]:


arr = np.arange(0,21)
arr1 = arr[arr%2==0]
arr2 = arr[arr%2!=0]


# In[201]:


arr1


# In[202]:


arr2


# In[204]:


arr=np.arange(30).reshape(2,3,5)
arr


# In[208]:


arr.flatten()


# In[219]:


arr.reshape(6,5).transpose()


# In[233]:


arr=np.arange(24)
arr


# In[234]:


arr.reshape(2,4,3)


# In[235]:


arr=np.arange(30).reshape(3,2,5)
arr


# In[240]:


cond=np.array(['a','b','c',], dtype='<U1')
cond


# In[248]:


arr[(cond=='a') | (cond=='c')]


# In[250]:


list_1=['a','b','c']
list_1=='b'


# In[251]:


cond=np.array(['a','b','c'])


# In[253]:


cond=='b'


# In[264]:


import numpy as np


# In[270]:


arr1=np.arange(8).reshape(2,-1)
arr2=np.arange(-40,40,10).reshape(2,-1)


# In[271]:


arr1


# In[272]:


arr2


# In[273]:


np.maximum(arr1,arr2)


# In[274]:


np.subtract(arr1,arr2)


# In[24]:


arr=np.array([[1,2],[3,4]])
arr


# In[25]:


arr.sum(axis=0)


# In[279]:


arr.sum()


# In[280]:


arr.mean()


# In[281]:


arr.var()


# In[282]:


arr.std()


# In[285]:


arr.argmin()


# In[287]:


arr.argmax()


# In[288]:


import numpy as np

identity=np.eye(4)
print(identity)


# In[291]:


x=np.arange(9).reshape(3,-1)


# In[296]:


print(x)


# In[298]:


np.diag(np.diag(x))


# In[1]:


import numpy as np

a=np.arange(4).reshape(-1,2)


# In[300]:


a


# In[301]:


print(a*a)


# In[303]:


print(a.dot(a))


# In[4]:


import pandas as pd


# In[5]:


obj = pd.Series([0,1,2,3,4,5,6,7], index=['a','b','c','d','e','f','g','h'], dtype=np.int64)


# In[6]:


obj


# In[7]:


obj[['e','c']]


# In[8]:


obj['e']


# In[10]:


obj['a':'c']


# In[20]:


import numpy as np


# In[29]:


import pandas as pd


# In[34]:


obj=pd.Series([2,1,3,3,1,np.nan,34,5])


# In[35]:


obj.unique() #중복을 자동으로 제거하지만 DataFrame 에서는 허용하지 않음.


# In[36]:


obj.value_counts(normalize = True)


# In[46]:


obj2=pd.Series(['a','b','c'], index=['i1','i2','i3'])
obj2


# In[32]:


obj = pd.Series([0,1,2,3,4,5,6,7], index=['a','b','c','d','e','f','g','h'], dtype=np.int64)
obj


# In[ ]:


obj.sort_index(ascending = True) #오름차순 list.sort(reverser= True, False)


# In[37]:


obj


# In[38]:


obj.sort_values(na_position='first')


# In[47]:


frame=pd.DataFrame(np.arange(9).reshape(3,-1), index=list('abc'), columns=list('def'))


# In[48]:


frame.sort_index()


# In[50]:


series=pd.Series([100,200,300])
series


# In[51]:


series.map({100:'C',200:'B',300:'A'})


# In[57]:


s= pd.Series([20,21,12],index=['London','New York','Helsinki'])
s


# In[58]:


def sub_custom_val(x,val):
    return x-val


# In[59]:


s.apply(sub_custom_val,args=(10,))


# In[60]:


args=np.array((10,))


# In[61]:


args=np.array(1)


# In[62]:


args.shape


# In[63]:


args.ndim


# In[65]:


s


# In[66]:


def add_custom_values(x,**kwargs): #**kwargs #**kw
    for month in kwargs:    #dict -> list 받게끔 loop
        x += kwargs[month]    #key에 접근
    return x


# In[68]:


s.apply(add_custom_values,june=30,july=20,auguest=25) #비효율적


# In[69]:


s.apply(add_custom_values,june=75) #Ln[68]과 똑같은 의미 month가 누적되기 때문


# In[72]:


frame=pd.DataFrame(np.arange(12).reshape(3,-1),
                    columns=['a','b','c','d'])


# In[73]:


frame


# In[74]:


frame.applymap(lambda x:x**2)


# In[105]:


frame=pd.DataFrame(np.arange(16).reshape(4,-1),
                    columns=['c1','c2','c3','c4'],
                    index=['r1','r2','r3','r4'])


# In[80]:


frame


# In[81]:


frame.drop('r1')


# In[82]:


frame


# In[83]:


frame.drop('c1', axis=1) #row로 접근되니깐 column으로 접근해주세요!


# In[85]:


frame.drop(['c1','c2'],axis=1) #column -> c1,c2에 있기 때문에 axis=1을 써야함.


# In[86]:


frame.drop(columns=['c1','c2']) #Ln(85)랑 같은 의미


# In[88]:


frame.drop(['r2'], inplace=True)
frame


# In[91]:


obj.isna().sum()


# In[92]:


obj


# In[93]:


obj.dropna() #리턴 안 됨


# In[95]:


obj


# In[112]:


frame.dropna(how='any') #row,col에 1개라도 있으면 삭제


# In[111]:


frame.dropna(how='all') #row,col에 다 있어야 삭제 됨.


# In[110]:


frame.fillna(method='bfill')


# In[109]:


frame.fillna({'0':10, '3':10})


# In[115]:


frame['r1:c3']='null'


# In[116]:


frame


# In[113]:


frame.fillna({'x1':10,'x3':5}) #feature -> string


# In[11]:


i=int(input())
for i in range(i,i+1):
    print('===',i,'단===')
    for j in range(1,10):
        print(i*j)


# In[12]:


x=[3,6,8,20,-7,5]


# In[13]:


print(x*10)


# In[23]:


i=int(input())
for i in range(i,i+1):
    print('~~',i,'단이지롱~~~')
    for j in range(1,10):
        print(i*j)


# In[24]:


word=["school","game","piano","science","hotel","mountain"]


# In[37]:


if len(word[i]) >= 6:
       print (word[i])


# In[33]:


a


# In[43]:


a=int(input())

for i in range(1,101):
    if i%15==0:
        print("3과 5의 공배수")
    elif i%5==0:
        print("5의 배수")
    elif i%3==0:
        print("3의 배수")
    else: 
        print("그냥 숫자")


# In[47]:


x=int(input())
s=True
a=sum(int(input()))


if s:
    print(a)


# In[1]:


import matplotlib.pyplot as plt


# In[2]:


get_ipython().run_line_magic('matplotlib', 'qt')
plt.plot([1,2,3,4,5,6])
plt.show()


# In[ ]:


plt.plot([1,3,5,110000,4234235690,2345205298])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




