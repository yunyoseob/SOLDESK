#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


get_ipython().run_line_magic('matplotlib', 'qt')
plt.plot([1,2,3,4,5,6])
plt.show()


# In[4]:


plt.plot([1,3,5,110000,4234235690,2345205298])
plt.show()
# plt.plot -> Line 2D 점만 찍으면 다 이어서 보여줌(자동)


# In[7]:


plt.plot([1,10,100,1000,500,800,1200,600,300,1000])
plt.show()


# In[22]:


import numpy as np


# In[34]:


x=np.arange(10)
y=x+10

plt.plot(x,y)
plt.show()


# In[35]:


#x축, y축을 건드려봄
plt.xlim([0,10])
plt.ylim([0,20])

plt.plot(x,y)
plt.show()


# In[47]:


x=np.linspace(-2,2,1000)
y=x**6+x**3+x**2+x*2+1

plt.plot(x,y)
plt.show()


# In[42]:


x=np.linspace(-2,2,1000)
y=x*3
z=x**3

plt.plot(x,y,z)
plt.show()


# In[48]:


x = np.linspace(-1.4, 1.4, 30)
plt.figure(1)
plt.subplot(211) #2행 1열 첫번째
plt.plot(x, x**2)
plt.title("Square and Cube")
plt.subplot(212) #2행 1열 두번째
plt.plot(x, x**3)
plt.figure(2, figsize=(10, 5))
plt.subplot(121)
plt.plot(x, x**4)
plt.title("y = x**4")
plt.subplot(122)
plt.plot(x, x**5)
plt.title("y = x**5")
plt.figure(1)      # 그림 1로 돌아가며, 활성화된 부분 그래프는 212 (하단)이 됩니다
plt.plot(x, -x**3, "r:")
plt.show()


# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=np.linspace(0,2*np.pi, 500)
################
y=np.sin(x)
z=np.cos(x)
# w=np.tan(x)
################
fig,ax=plt.subplots()

ax.plot(x,y,label='sin', color=(0.9999,0.1111,0.5555))
# R:0.1, G: 0.3 B: 0.5만큼 적용해서 그려줘
# image,plot -> R G B

ax.plot(x,z,label='cos', color=(0.1111,0.9999,0.5555))
# ax.plot(x,w,label='tan')

ax.legend()
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as plt
data = {'사과':21, '바나나':15, '배':5, '키위':20)

names = list(data.keys())
values=list(data.)


# In[65]:


data=np.random.rand(100)
fig,ax=plt.subplots()
ax.hist(data,bins=100, facecolor='b', color=(0.9999,0.1111,0.5555))
plt.show()


# In[69]:


# Scatter plot: 뿌려놓은 것처럼 생김.
np.random.seed(7877)

n=50
x=np.random.rand(n)
y=np.random.rand(n)

plt.scatter(x,y)
plt.show()


# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratio=[34,32,16,18]

labels=['apple','banana','melon','grapes']

plt.pie(ratio, labels=labels,autopct='%1f%%')
plt.show()


# In[77]:


import seaborn as sns


# In[ ]:





# In[ ]:





# In[ ]:




