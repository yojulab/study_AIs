#!/usr/bin/env python
# coding: utf-8

# ### 지도학습(Supervised Learing)
# - 목표변수(target, Y) 있는 학습법

# In[19]:


import pandas as pd


# In[20]:


df_BCD = pd.read_csv('datasets/BreastCancerWisconsinDataSet.csv')
df_BCD[:2]


# #### 목표변수 - 연속형
# - 목표변수 : radius_mean
# - 설명변수 : drop columns - radius_mean, id, diagnosis, Unnamed: 32

# In[21]:


df_BCD.info()


# #### PreProcessing

# ##### 목표변수와 설명변수 축출

# In[22]:


#df_BCD.drop(columns=['id', 'diagnosis', 'Unnamed: 32']).info()
df_BCD_extract = df_BCD.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])


# In[23]:


df_BCD_extract.isnull().sum()


# ##### Structured data 
# - 목표변수, Y, **Target**
# - 설명변수, X, **Label**

# In[24]:


target = df_BCD_extract['radius_mean']
labels = df_BCD_extract.drop(columns=['radius_mean'])


# In[25]:


#target[:2]
# labels[:2]
target.shape, labels.shape


# #### 모델(알고리즘)학습

# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


model = LinearRegression()


# ##### 모델 학습

# In[28]:


model.fit(labels, target)


# #### 평가 (나중에 함)

# #### 미래예측(서비스 개시)

# In[29]:


test_set = labels[:1]
test_set


# In[30]:


model.predict(test_set)


# In[31]:


target[:1]


# In[ ]:


# 모든 컬럼에 대해 값들을 무작위로 삭제
delete_ratio =0.2 #삭제 비율(50%)

