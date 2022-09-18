# 20210409 머신러닝

# 20210413 이중중첩문 문제 나옴 (기출변형)
# 20210413 class에서는 나오지 않음

import statsmodels as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import datasets
from sklearn.preprocessing import PolynomialFeatures


# 20210413 시험 관련

# github 2~4번 문제 안나옴
# 로또 번호 자동 생성기 문제 안나옴
# github 비트연산 안나옴
# github 21번 문제 안 나옴
# 패키지 쓰면 감점

# 회귀분석 복습
# 다중 공산성: 독립변수끼리 상관관계가 높으면서 나타나는 문제
# f(ax+by)=af(X)+bf(y)


# 독립변수 하나일 때, SSE
# 독립변수 두 개 일 때, LSE
# Gradient Descent에서 다음달에 한 문제 나옴

# min||y-Xb||^2 에서 값이 튀지 않도록 arg를 통해 설정
# arg min ||y-Xb||^2 + lambda||b||^2 #여기서 lambda||b||^2가 ridge이다. #L2 규제를 추가한 방식

# 경사하강법(Gradient Descent)
# 경사하강법 매우 중요
# y(hat) = x0 + w1x1 + w1x1 +...+ wnxn 을 미분을 통해 구한다.
# 경사하강법 수행 프로세스 2/N -> 개수만큼 나눴다. 별로 안 중요함.
# MSE = 1/n-2 * SSE # n-2는 자유도이므로 큰 의미는 없음

# 신뢰구간을 알 수 있을 때만 p-value 등등을 쓸 수 있다.

# 도구 불러오기
import statsmodels as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import datasets
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

# 히트맵을 통해 상관관계를 시각화 한 다음 데이터를 넣고 뺌으로써
# R-Square를 통해 증명 
# 결측치 넣고 빼고 하면서 R-square 값의 움직임을 봐야함.
# 이 작업을 반복해야함.


## df.to_csv("somename.csv")로 하면 한 번에 파일을 불러올 수 있음.


#2021 04 12 수업
# encoding  할 때 한 번에 하지 말고 각각 해야함.



















































