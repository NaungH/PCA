#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in the data and perform basic exploratory analysis

# In[5]:


df = pd.read_csv('C:/Users/Naung Naung/OneDrive/Desktop/WAREM/Hiwi/Sorted_Data_Inn.csv')
df.head(10)


# #### Basic statistics

# In[6]:


df.iloc[:,1:].describe()


# #### Boxplots by output labels/classes

# In[7]:


for c in df.columns[1:]:
    df.boxplot(c,by='Dataset',figsize=(7,4),fontsize=14)
    plt.title("{}\n".format(c),fontsize=16)
    plt.xlabel("Dataset", fontsize=16)


# In[8]:


plt.figure(figsize=(10,6))
plt.scatter(df['Temp_up20'],df['kf_up20'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Scatter plot of Temperature and Kf",fontsize=15)
plt.xlabel("Temp_up20",fontsize=15)
plt.ylabel("kf_up20",fontsize=15)
plt.show()


# In[9]:


plt.figure(figsize=(10,6))
plt.scatter(df['Temp_up20'],df['Fines'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Scatter plot of Temperature and Fine Sediment",fontsize=15)
plt.xlabel("Temp_up20",fontsize=15)
plt.ylabel("Fines",fontsize=15)
plt.show()


# #### Are the features independent? Plot co-variance matrix
# 
# It can be seen that there are some good amount of correlation between features i.e. they are not independent of each other, as assumed in Naive Bayes technique. However, we will still go ahead and apply yhe classifier to see its performance.

# In[10]:


plt.figure(figsize=(10,6))
plt.scatter(df['kf_up20'],df['Fines'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Scatter plot of Kf and Fine Sediment",fontsize=15)
plt.xlabel("kf-up20",fontsize=15)
plt.ylabel("Fines",fontsize=15)
plt.show()


# In[11]:


plt.figure(figsize=(10,6))
plt.scatter(df['Por'],df['Fines'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Scatter plot of Porosity and Fine Sediment",fontsize=15)
plt.xlabel("Por",fontsize=15)
plt.ylabel("Fines",fontsize=15)
plt.show()


# In[88]:


def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Wine data set features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)


# In[12]:


matrix = df.corr()
print("Correlation Matrix is : ")
print(matrix)


# In[14]:


import os
import pandas as pd
import numpy as np
import seaborn as sn
 
Dataset = pd.read_csv('C:/Users/Naung Naung/OneDrive/Desktop/WAREM/Hiwi/Sorted_Data_Inn.csv')

numeric_col = ['Temp_up20','Temp_low20','Ox_up20','kf_up20','kf_low20','d10','dm','d84','Cu','Cc','Por','Fines']
 
corr_matrix = Dataset.loc[:,numeric_col].corr()
print(corr_matrix)

sn.heatmap(corr_matrix, annot=True)


# ## Principal Component Analysis

# ### Data scaling
# PCA requires scaling/normalization of the data to work properly

# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler = StandardScaler()


# In[17]:


X = df.drop('Dataset',axis=1)
y = df['Dataset']


# In[18]:


X = scaler.fit_transform(X)


# In[19]:


dfx = pd.DataFrame(data=X,columns=df.columns[1:])


# In[20]:


dfx.head(10)


# In[21]:


dfx.describe()


# ### PCA class import and analysis

# In[3]:


from sklearn.decomposition import PCA


# In[4]:


pca = PCA(n_components=None)


# In[5]:


dfx_pca = pca.fit(dfx)


# #### Plot the _explained variance ratio_

# In[6]:


plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
            y=dfx_pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
plt.xlabel("Principal components",fontsize=15)
plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()


# **The above plot means that the $1^{st}$ principal component explains about 36% of the total variance in the data and the $2^{nd}$ component explians further 20%. Therefore, if we just consider first two components, they together explain 56% of the total variance.**

# ### Showing better class separation using principal components

# #### Transform the scaled data set using the fitted PCA object

# In[26]:


dfx_trans = pca.transform(dfx)


# #### Put it in a data frame

# In[27]:


dfx_trans = pd.DataFrame(data=dfx_trans)
dfx_trans.head(10)


# #### Plot the first two columns of this transformed data set with the color set to original ground truth class label

# In[2]:


plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[1],dfx_trans[2],c=df['Temp_up20'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()


# In[35]:


df = pd.read_csv('C:/Users/Naung Naung/OneDrive/Desktop/WAREM/Hiwi/Sorted_Data_Inn.csv')
df.head(10)


# In[65]:


from sklearn.preprocessing import StandardScaler
inndata = ['Temp_up20','Temp_low20','Ox_up20','kf_up20','kf_low20','d10','dm','d84','Cu','Cc','Por','Fines']
# Separating out the features
x = df.loc[:, inndata].values
# Separating out the target
y = df.loc[:,['Dataset']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x)


# In[62]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[63]:


finalDf = pd.concat([principalDf, df[['Dataset']]], axis = 1)


# In[64]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
Dataset = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for Dataset, color in zip(Dataset,colors):
    indicesToKeep = finalDf['Dataset'] == Dataset
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(Dataset)
ax.grid()


# In[ ]:




