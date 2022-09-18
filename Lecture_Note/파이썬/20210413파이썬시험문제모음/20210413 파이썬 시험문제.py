# 20210413 파이썬 시험문제
# 1번 문제
Apart=[[101,102,103,104],[201,202,203,204],[301,302,303,304],[401,402,403,404]]
Unpaid=[101,204,302,402]

for floor in Apart:
    for house in floor:
        if house in Unpaid:
            print('우유 배달해야 하지 말아야 할곳', house)
            break
        else:
            print('우유 배달해야 할 곳', house)

# 2번 문제
def add_all(*inputs):
    s=0
    for i in inputs:
        for j in i:
            s+=j
        return s

add_all([1,2,3,4,5])
add_all((1,2,3,4,5))

# 3번 문제 # 내 버전
def solution(n):
    new = str(n)
    add = 0
    for i in range(len(new)):
        add += int(new[i])
    return add
print(solution(123))

# 4번 문제 #정답
x = int(input("Enter:"))
a=""
while x:
    a += str(x%2)
    x = int(x/2)
print(a[::-1])

# 5번 문제
names=['홍길동','홍계월','김철수','이영희','박첨지']
for name in names:
    print(name[0]+'*'+name[-1])







