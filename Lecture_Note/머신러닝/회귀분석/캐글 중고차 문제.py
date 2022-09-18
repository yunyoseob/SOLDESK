print('hello')

# 캐글 중고차 문제

# 1. 도구 불러오기
import statsmodels as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import datasets
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns


# 2. 파일 불러오기
audi=pd.read_csv('C:/Users/sundooedu/Desktop/archive/audi.csv')
bmw=pd.read_csv('C:/Users/sundooedu/Desktop/archive/bmw.csv')
cclass=pd.read_csv('C:/Users/sundooedu/Desktop/archive/cclass.csv')
focus=pd.read_csv('C:/Users/sundooedu/Desktop/archive/focus.csv')
ford=pd.read_csv('C:/Users/sundooedu/Desktop/archive/ford.csv')
hyundai=pd.read_csv('C:/Users/sundooedu/Desktop/archive/hyundi.csv')
merc=pd.read_csv('C:/Users/sundooedu/Desktop/archive/merc.csv')
skoda=pd.read_csv('C:/Users/sundooedu/Desktop/archive/skoda.csv')
toyota=pd.read_csv('C:/Users/sundooedu/Desktop/archive/toyota.csv')
uncleancclass=pd.read_csv('C:/Users/sundooedu/Desktop/archive/unclean cclass.csv')
uncleanfocus=pd.read_csv('C:/Users/sundooedu/Desktop/archive/unclean focus.csv')
vauxhall=pd.read_csv('C:/Users/sundooedu/Desktop/archive/vauxhall.csv')
data_vw=pd.read_csv('C:/Users/sundooedu/Desktop/archive/vw.csv')

# 
data_vw.describe()
len(data_vw)

# 
print(data_vw['model'].value_counts()/len(data_vw))
sns.countplot(x='model', data=data_vw)
plt.show()

sns.pairplot(data_vw) #3차원이상 데이터 추출
plt.show()

#
data_vw_expanded


## Gradient Descent 설명
# https://github.com/Youngpyoryu/SD_academy/blob/6e4078e14ddc489bbea6bc95d03d00ac8af2ccc4/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%ED%9A%8C%EA%B7%80(Regression)/Regression_210409.ipynb


#################### 선생님 풀이###########################
#### 각종 차트 보기 ####
data_vw.describe()

sns.countplot(data_vw['transmission'])
plt.show()

print(data_vw['model'].value_counts() / len(data_vw))
sns.countplot(y = data_vw["model"])
plt.show()

sns.countplot(data_vw['fuelType'])
plt.show()


sns.countplot(y = data_vw['year'])
plt.show()


plt.figure(figsize=(15,5))
sns.barplot(x = data_vw['year'], y = data_vw['price'])
plt.show()

sns.barplot( x= data_vw['transmission'], y= data_vw["price"])
plt.show()

plt.figure(figsize=(15,10),facecolor='w') 
sns.scatterplot(data_vw["mileage"], data_vw["price"], hue = data_vw["year"])
plt.show()

#### pre-processing for modeling ####
data_vw_expanded = pd.get_dummies(data_vw)
## 데이터를 data_vw를 get_dummies를 통해 변형
## get_dummies: 명목형 변수를 one-hot encoding을 해줌
data_vw_expanded.head()

std = StandardScaler()
# StandardScaler(정규화) : z = (x-u)/s

data_vw_expanded_std = std.fit_transform(data_vw_expanded)
data_vw_expanded_std = pd.DataFrame(data_vw_expanded_std, columns = data_vw_expanded.columns)
print(data_vw_expanded_std.shape)
data_vw_expanded_std.head()

##

from sklearn.feature_selection import SelectKBest, f_regression
#SelectKBest 모듈은 target 변수와 그외 변수 사이의 상관관계를 계산하여 가장 상관관계가 높은 변수 k개를 선정할 수 있는 모듈입니다
# f_regression 참고 : (https://woolulu.tistory.com/63)
#Linear model for testing the individual effect of each of many regressors. 
#This is a scoring function to be used in a feature selection procedure, not a free standing feature selection procedure.

column_names = data_vw_expanded.drop(columns = ['price']).columns

no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 40, 2):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train, y_train)
    X_test_transformed = selector.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared_train.append(regressor.score(X_train_transformed, y_train))
    r_squared_test.append(regressor.score(X_test_transformed, y_test))
    
sns.lineplot(x = no_of_features, y = r_squared_train, legend = 'full')
sns.lineplot(x = no_of_features, y = r_squared_test, legend = 'full')

plt.show()

##
selector = SelectKBest(f_regression, k = 23)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)
column_names[selector.get_support()]

##
def regression_model(model):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score

##
model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)

model_performance

##
regressor = sm.OLS(y_train, X_train).fit()
print(regressor.summary())

X_train_dropped = X_train.copy()


##
while True:
    if max(regressor.pvalues) > 0.05:
        drop_variable = regressor.pvalues[regressor.pvalues == max(regressor.pvalues)]
        print("Dropping " + drop_variable.index[0] + " and running regression again because pvalue is: " + str(drop_variable[0]))
        X_train_dropped = X_train_dropped.drop(columns = [drop_variable.index[0]])
        regressor = sm.OLS(y_train, X_train_dropped).fit()
    else:
        print("All p values less than 0.05")
        break

print(regressor.summary())

##
poly = PolynomialFeatures()
X_train_transformed_poly = poly.fit_transform(X_train)
X_test_transformed_poly = poly.transform(X_test)

print(X_train_transformed_poly.shape)

no_of_features = []
r_squared = []

for k in range(10, 277, 5):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared.append(regressor.score(X_train_transformed, y_train))
    
sns.lineplot(x = no_of_features, y = r_squared)

##
selector = SelectKBest(f_regression, k = 110)
X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)
X_test_transformed = selector.transform(X_test_transformed_poly)

##
models_to_evaluate = [LinearRegression(), Ridge(), Lasso()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Polynomial","Model": model, "Score": score}, ignore_index=True)

model_performance









































