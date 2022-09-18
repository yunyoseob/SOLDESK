# R 실습 20210614

1+1

a<-c(1,2,3) #combination약자 c
var2 <- c(1:5)
var3 <- c(6:10)
var2+var3 # 두 combination끼리 더할 수도 있음.

var1<-seq(1,70,by=2) 
# 1부터 70까지 두 칸씩 띄워서 저장
var1

var1=seq(1,70,by=2) # =도 된다.
var1

# 조건문과 비교,논리 연산자
a=3
b=5

if(a>b) {print(a)
}else {print(b)}

# 반복문
i<-0
while(i<10){
  i<-i+1
}

# for문
sum<-0
for (i in c(1,4,7)){
  sum<-sum+i
}

# for+if 문
sum<-0
for (i in 5:15){
  if(i%%2==0)
  {
    next;
  }
  if(i%%10==0)
  {
    break;
  }
  sum<-sum+i
}

# 함수
f<-function(n){
  sum<-0;
  for (i in 1:n){
    sum<-sum+i
  }
  return(sum)
}

f(12)

# 함수 동일적용
g<-f
g(12)

#(편)미분
f<-expression(2*x^3-y*x^2+2*y^2+1)
D(f, "x") # x에 대해 편미분
D(f, "y") # y에 대해 편미분

# 정적분
f<-function(x) 2*x^3-3*x^2+1
integrate(f,0,3)

# 기초적인 시각화 도구
hist(c(5,1,6,4,3,2,85,7,1,5,63))


# tool 사용해보기(ggplot2)
install.packages('ggplot2')

library(ggplot2)

x<-c('a','a','b','c')
qplot(x)

# highwqy
qplot(data=mpg, x=hwy)

#x축에 cty
qplot(data=mpg, x=cty)

#x축에 drv, y축은 hwy, 선 그래프 형태
qplot(data=mpg, x='drv', y='hwy',geom='line')

# x축에 drv, y축을 hwy, box plot 형태
qplot(data=mpg, x='drv', y='hwy', geom='boxplot')

?qplot # 설명서 보기

english <-c(90,80,60,70)
english

math<-c(50,60,100,20)

class<-c(1,1,2,2)

df_midterm<-data.frame(english,math,class) # 데이터 프레임 만들기 
df_midterm

#english math class
#1      90   50     1
#2      80   60     1
#3      60  100     2
#4      70   20     2

mean(df_midterm$english)
mean(df_midterm$math)

# CSV 파일 불러오기
install.packages('readxl')
library(readxl)

df_exam<-read_excel('C:/Users/sundooedu/Desktop/R실습/excel_exam.xlsx')
df_exam

mean(df_exam$math)
mean(df_exam$english)

head(df_exam, 5) #위에 다섯 개 행만 출력
View(df_exam) #새로운 탭으로 보기

str(df_exam) #데이터의 속성을 보여줌
summary(df_exam) # 파이썬의 describe 같음

sd(df_exam$id) #표준편차 보기

df_new<-data.frame(var1=c(1,2,1),
                   var2=c(2,3,2))
df_new

install.packages('dplyr') #테이블을 가공하는 패키지
library(dplyr) #rename으로 테이블 이름을 바꿔주기 위해 패키지를 불러옴


df_naw<-df_new #copy 하기
df_naw

df_naw<- rename(df_naw, v2=var2)
df_naw

df_naw$var_sum <-df_naw$var1+df_naw$v2 #파생변수
df_naw$var_sum

df_naw

mpg$total<-(mpg$cty+mpg$hwy)/2
head(mpg)
summary(mpg$total)
hist(mpg$total)

mpg$test <-ifelse(mpg$total >=20, 'pass','fail')
head(mpg,20)

table(mpg$test)

#막대그래프로 빈도 표현하기.
library(ggplot2)
qplot(mpg$test)

#중첩 조건문을 사용 total 30보다 크면 grade A ----
mpg$grade <- ifelse(mpg$total >=30, 'A',
                    ifelse(mpg$total >=20, 'B','C' ))

mpg$grade

# 테이블 만들기
table(mpg$grade)
#A   B   C 
#10 118 106 

qplot(mpg$grade)


