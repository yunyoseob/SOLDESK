import pandas as pd
from konlpy.tag import Okt

okt=Okt()

txt='아버지 가방에 들어가신다.'

okt.pos(txt)


from konlpy.corpus import kolaw
from konlpy.tag import OKt
import nltk

doc_ko=open("C:Users/sundooedu/Desktop/애국가.txt", "r")

f=open()

#!wget -nc -q https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt

#wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt

ratings_train=pd.read_csv('C:/Users/sundooedu/Desktop/ratings_train.txt', error_bad_lines=False)









