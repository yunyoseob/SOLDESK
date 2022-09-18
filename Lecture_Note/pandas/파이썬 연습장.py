print("hello world")


dict_3={'홍길동':100, '홍계월':200, '슈퍼맨':150, '베트맨':250}
print(dict_3.keys())
print(dict_3.values())

set_5={1,2,3,4,}
set_6={4,5,6,7}
set_5.intersection(set_6)
set_5.union(set_6)

set_5={4,5,6,7}
set_5.update([8,9])
set_5

a=input(int())

a='강아지'
b='고양이'
print('%s와 %s를 더하면 개냥이' %(a,b,))

a=3
b=5
a<b

e=[1,3,5,7]
0 in e

money= 1000
card = False

if card:
    if money <30000:
        print("삼겹살을 먹는다")
    else:
        print("소고기를 먹는다")
else:
    if money <= 1000:
            pass
    else:
        print("라면을 먹는다")

value=20
is_even=True if value %2==0 else False
print(is_even)
True

num=1
print('문자열 포맷팅 하기 예시 %d' %num)

s='이름'
print('내이름은 %s 에요' %s)

pie=3.141592
print('파이의 소숫점 첫째자리에서 반올림 하면 %0.123f 이다' %pie)

a='pen'
b='pineapple'

print('%s과 %s를 더 하면 %s' %(a,b,a+b))

print('{가}와 {나}를 더하면 {다}다.'.format(가='펜', 나='파인애플', 다='펜파인애플'))

year=2021
f'올해는 {year}'

drink='핫식스'
nums=2

f'나는 오늘 {drink}를 {nums}캔이나 마셨다.'


i=1

while i <= 10: 
    if i%2 == 0: 
        print(i) 
    else:
        i+=1

i=90
while i:
    i+=1
    if i == 100:
        print("축하합니다. %d번째 방문자입니다." %i)
        break
    print("감사합니다. 이벤트가 종료되었습니다.")

i=0

while i < 11:
    i += 1
    if i == 6:
        continue
    if i%2 ==0:
        print(i)


for i in range(1,100,10):
    print(i)        

# 1~100까지 10의 배수만을 출력하라.
i=1

while i<100:
    i+=1
    if i%10 == 0:
        print(i)


# 구구단 18~20단 출력
for i in range(18,21):
    print('===',i,'단===')
    for j in range(1,10):
    print(i*j)

# 함수
def add(a,b):
    result=a+b
    return result

c= add(3,5)
c

def sub(a,b):
    print('뺄셈의 결과는 %d입니다.' %(a-b))
    return

d= sub(1,2)

def gugudan(num):
    for i in range(1,10):
        print(f'{num}*{i}={num*i}')

gugudan(3)


a=156
b=43

def sub(a,b):
    print('나눗셈의 결과는 %d입니다.' %(a//b))
    return
print(sub(a,b))






























































