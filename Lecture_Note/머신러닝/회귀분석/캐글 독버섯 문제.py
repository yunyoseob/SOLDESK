print("Hello")
# 도구 불러오기
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

# 파일 불러오기
mushrooms=pd.read_csv('C:/Users/sundooedu/Desktop/archive/mushrooms.csv')
mushrooms.describe()

#특징보기
mushrooms.head()








































































































