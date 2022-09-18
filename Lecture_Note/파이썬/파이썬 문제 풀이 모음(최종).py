# 파이썬 문제 풀이 모음

# 파이썬PDF 연습문제1 1번 p.44
# 10의 제곱을 출력해보자
10**2
10*10


# 파이썬PDF 연습문제2-1 1번 p.56
# 사과가 5개, 오렌지가 3개 있을 때 총 과일의 갯수를 구해보자
# 사과는 apple이라는 변수, 오렌지는 orange라는 변수에 할당 한 후,
# 총 과일의 갯수를 total이라는 변수에 저장해보자.
apple=5
orange=3
total=apple+orange
print(total)

# 파이썬PDF 연습문제2-2 2번 p.57
# 국어는 100점, 영어는 88점, 수학은 94점 일 때, 평균을 구하려고 한다.
# 각각의 점수를 kor, eng, math라는 변수에 저장한 후 평균을 구해 avg라는 변수에 할당해보자
kor=100
eng=88
math=94
avg=(kor+eng+math)/3
print(avg)

# 파이썬PDF 연습문제 quiz p.62
# a,b의 정수형의 합
a=5.3
b=5.7
a=int(a)
b=int(b)
print(a+b)

# 파이썬PDF 연습문제 quiz p.79
# str_7 = 'Alphabet' 이라고 할 때, 모든 것을 소문자로 바꾼다음에 a문자의 개수를 세라
str_7 = 'Alphabet'
str_7.lower().count('a')

# 파이썬PDF 연습문제3 1번 p.107
# '아메리카노를 많이많이많이 좋아한다.'를 곱하기와 더하기를 써서 출력하세요.
a='아메리카노를'
b='많이'
c='좋아한다'

a+(b*3)+c

# 파이썬PDF 연습문제3 2번 p.108
# 좋아하는 음식 5개를 food라는 리스트에 담은 후 리스트 가장 앞에 파이썬, 가장 뒤에는
# 집이라는 단어가 오도록 수정한 후, pop을 이용하여 파이썬을 삭제하고, remove를 이용하여
# 집을 삭제해보세요.

food=['사과','배','오렌지','삼겹살','회']
food.append('집')
food.insert(0,'파이썬')
food
food.remove('집')
food.pop(0)
food

# 파이썬PDF 연습문제4 + 연습문제5-2 p.127, p.153
# 사용자로 점수를 3개를 입력받아 모든 점수가 65보다 클 경우 합격,
# 아닐경우 불합격을 출력하세요.
# 0~100점이 아니면 잘못된 점수가 입력되었습니다를 출력할 것.

a=int(input("점수를 입력하세요."))
b=int(input("점수를 입력하세요."))
c=int(input("점수를 입력하세요."))

if 100 >= a > 65 and 100 >= b >65 and 100 >= c >65:
    print('합격')
elif 0<a<=65 or 0<b<=65 or 0<c<=65:
    print('불합격')
else:
    print('잘못된 정보가 입력되었습니다.')

# 파이썬PDF 연습문제4-1번 p.128
# 국어, 영어, 수학 점수가 키로 하는 딕셔너리를 만들어보자.
# 각각의 점수는 국어가 87점, 영어가 88점, 수학이 92점이다.

dic_1={'국어':87, '영어':88, '수학':92}

# 파이썬PDF 연습문제4-2번 p.128 다음 연산들의 값을 예측해보자
3 <= 1 #False
6%3==0 #True
s1={'a','b','c'}
s2={'b','e','f','g'}

s1&s2 # 교집합
s1-s2 # 차집합(왼쪽)
s2-s1 # 차집합(오른쪽)

# 파이썬PDF 연습문제 p.129
# a="20190505chicken19000" 은 잘못 쓰여진 변수이다.
# 2019는 year라는 변수에, 그 다음 0505를 day라는 변수에,
# chicken을 menu라는 변수에, 19000을 money라는 변수에 인덱싱을 사용하여
# 각각 저장하고 출력하시오.

a="20190505chicken19000"
year=a[0:4]
year
day=a[4:8]
day
menu=a[8:15]
menu
money=a[15:]
money

# 파이썬PDF 연습문제5-1 p.152 b로 a를 나눈 나머지가 3 초과면 실패,
# 3이면 무승부, 3미만이면 성공이 출력되도록 만들어보자.

# 내 풀이 방식
a=int(input("a="))
b=int(input("b="))

if a%b >3:
    print('실패')
elif a%b==0 :
    print('무승부')
else:
    print('성공')

# 파이썬PDF 연습문제 5-3 p.154 사용자로부터 정수를 하나 입력받아 입력한 정수가 홀수인지
# 짝수인지 판별하여라.
# 0은 짝수로 취급한다.

a=int(input("정수를 입력하세요."))

if a%2 ==0:
    print("짝수")
elif a==0:
    print("짝수")
else:
    print("홀수")

# 파이썬PDF 연습문제 5-4 p.155 
# 양의 정수 하나를 입력받아 이 정수가 2의 배수인지 3의 배수인지 작성하시오.

# 내 풀이 0 이하 정수에 문제 있음.
a=int(input("정수를 입력하세요."))

if a%2 ==0:
    print("2의 배수")
elif a%3 ==0:
    print("3의 배수")
elif a<0:
    print("잘못 입력하셨습니다.")
else:
    print("2의 배수도 3의 배수도 아닙니다.")


# 선생님 풀이
num=int(input("정수를 하나 입력하세요."))

if (num%2) == 0:
    if (num%3) == 3:
        print('2와 3으로 나누어 집니다.')
    else:
        print('2로 나누어 집니다.')
elif (num%3) == 0:
    print('3으로 나누어 집니다.')
else:
    print('어느 것으로도 나누어 지지 않습니다.')

# 파이썬PDF 연습문제 5-4 p.170
# 1부터 10까지 합을 range 함수를 이용하여 구하시오.

# 내 풀이
i=0
sum=0

while i<11:
    if i==10:
        break
    i+=1
    sum=sum+i
    print(sum)

# 선생님 풀이
def sum_num(n):
    s=0
    for i in range(1,n+1):
        s=s+i
    return s
print(sum_num(10))


# 선생님 또 다른 풀이
i=0
sum=0

for i in range(0,11):
    sum=sum+i
    i+=1
    print(sum)

# p.203 구구단 만들기
#1
for i in range(1,10):
    print(f'=== {i} 단===')
    for j in range(1,10):
        print(i*j)

#2

def gugu(num):
    for i in range(1,10):
        print(f'{num}*{i}={num*i}')

gugu(3)

# *args 문제!! 시험에 나옴.
def add_all1(*args):
    s=0
    for i in args:
        for j in i:
            s+=j
        return s

add_all1([1,2,3,4,5,6,7,8,9,10])

# 파이썬 심화 p. 220
# 사람들에게 먼저 온 순서대로 번호표를 나누어 주는 함수를 작성해보자.

people=['펭수','뽀로로','뚝딱이','텔레토비']

def func1(line):
    new_lines=[]
    i=1
    for x in line:
        print('대기번호 %d번: %s' %(i,x))
        new_lines.append((i,x))
        i+=1
    return new_lines

lines=func1(people)


# enumerate를 이용하여 풀기
people=['펭수','뽀로로','뚝딱이','텔레토비']

def func1_with_enu(line):
    new_lines=[]
    for idx, val in enumerate(line):
        print("대기번호 %d번: %s" %(idx+1, val))
    return new_lines

lines=func1_with_enu(people)

# lambda 활용
items = [1,2,3,4,5]
str_items=list(map(lambda x: str(x), items))
print(str_items)

# list comprehension 구구단 2단 출력 p.236
list_1=[ x**2 for x in range(1,11) if x**2<50 ]
list_1

# list comprehension p.236
list_2="코로나 바이러스를 예방하기 위해 사회적 거리두기를 실천합시다. 마스크를 끼고 손씻기를 생활화 합시다."
list_2=[s for s in list_2.split() if len(s)<5]
list_2

# git hub 2,4,9,10번,15번,21번 문제에서는 안 나옴.
# git hub 문제 
# https://github.com/Youngpyoryu/SD_academy/blob/b6d31542847ae504515027e0c82c44ac4850963f/%ED%8C%8C%EC%9D%B4%EC%8D%AC/%ED%8C%8C%EC%9D%B4%EC%8D%AC_%EB%AC%B8%EC%A0%9C.ipynb

# git hub 1번 문제 (비식별화문제)
names=['홍길동','홍계월','김철수','이영희','박첨지']
for name in names:
    print(name[0]+'*'+name[-1])

# git hub 5번 문제(복리이자율문제)
year=0
money=1000

while money<2000:
    year+=1
    money+=money*0.07
    
    print(year)

# git hub 6번 문제(모음제거문제)
def anti_vowel(text):
    return "".join([x for x in text if x not in "aeiouAEIOU"])

anti_vowel("Life is too short, you need python!")

# git hub 7번 문제(리스트 중에서 홀수에만 2를 곱하여 저장하는 코드 작성)
lc_1=[2*x for x in range(0,11) if x%2 == 1 ]
lc_1

# git hub 8번 문제(list comprehension을 용해서 lc_1이 1부터 100사이의 8의 배수 출력)
lc_2=[x for x in range(1,101) if x%8 == 0]
lc_2


# git hub 11번 문제(행렬 덧셈)
#https://velog.io/@joygoround/test-%ED%96%89%EB%A0%AC%EC%9D%98-%EB%8D%A7%EC%85%88-%ED%8C%8C%EC%9D%B4%EC%8D%AC

# zip 안 쓰고 # ???
arr1=[[1,2],[2,3]]
arr2=[[3,4],[5,6]]


# zip활용
arr1=[[1,2],[2,3]]
arr2=[[3,4],[5,6]]

def solution2(arr1, arr2):
    answer2=[[c+d for c, d in zip(a,b)] for a,b in zip(arr1,arr2)]
    return answer2

solution2(arr1, arr2)

# git hub 12번 문제(소수 개수 반환하는 함수)
a=int(input())

def solution(num):
    indicator = 0
    for i in range(2, int(num)):
        if int(num) % i == 0:
            indicator = 1
    if indicator == 0:
        return True

result = []
for i in range(2,a):
    if solution(i):
        result.append(i)

print('Total : {}'.format(len(result)))

# github 16번 문제(비트연산 문제)
n=int(input('숫자 n을 입력하세요.'))

count=1

for i in range(n-1,-1,-1):
    print('O'*i+'X'*count)
    count=count+1

# github 17번 문제(조건 입력값 문제)
# https://codingdojang.com/scode/400?orderby=time&langby=cpp <해답>

# github 18번 문제(콤마가 포함된 금액 표기식 문자열로 바꾸어주는 프로그램 작성)
a=int(input())

b= format(a, ',')

print(b)


# github 19번 문제(약수의 개수)
n=int(input("약수의 개수를 입력하세요."))
num=[]

for i in range(1,n+1):
    if n%i == 0:
        print("{%d}" %i)
        num.append(n)
    else:
        i +=1

print("약수의 총 개수는 %d개 입니다," %len(num))


# github 20번 문제 # 자릿수 더하기
n=2**(int(input()))

def solution(n):
    new = str(n)
    add = 0
    for i in range(len(new)):
        add += int(new[i])
    return add
print(solution(n))

# github 22번 문제 # 2진법 출력
value=int(input())
b=bin(value)
print(value, b)

#github 23번 문제 #가성비 최대화
orp = 10
org = 150
adp = 3
adg = [30, 70, 15, 40, 65]
adg.sort(reverse=True)

for i in adg:
    if org / orp > (org + i) / (orp + adp):
        break
    else:
        org += i
        orp += adp

print(org / orp)







































