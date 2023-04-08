#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


dt = load_iris()


# In[3]:


dir(dt)


# In[4]:


dt.data


# In[5]:


import pandas as pd
dt_df = pd.DataFrame(dt.data)


# In[6]:


dt_df


# In[7]:


dt_df.columns = dt.feature_names
dt_df


# In[8]:


dt_df2 = dt_df.drop(['petal length (cm)','petal width (cm)'], axis=1)
dt_df2


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(dt_df2[['sepal length (cm)']])
dt_df2['sepal length (cm)'] = scaler.transform(dt_df2[['sepal length (cm)']])

scaler.fit(dt_df2[['sepal width (cm)']])
dt_df2['sepal width (cm)'] = scaler.transform(dt_df2[['sepal width (cm)']])

dt_df2


# In[10]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[11]:


plt.scatter(dt_df2['sepal length (cm)'],dt_df2['sepal width (cm)'])


# In[12]:


km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(dt_df2)
y_predicted


# In[13]:


dt_df2['cluster']= y_predicted
dt_df2.head()


# In[14]:


km.cluster_centers_


# In[15]:


df1 = dt_df2[dt_df2.cluster==0]
df2 = dt_df2[dt_df2.cluster==1]
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='green')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='red')


# In[16]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(dt_df2)
    sse.append(km.inertia_)


# In[17]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

