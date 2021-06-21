#!/usr/bin/env python
# coding: utf-8

# In[43]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score


# In[44]:


url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
student=pd.read_csv(url)


# In[6]:


student


# In[7]:


student.shape


# In[11]:


student.size


# In[12]:


student.describe()


# In[13]:


student.info()


# In[14]:


#visualisation


# In[19]:


student.hist(figsize=(10,10))
plt.show()


# In[18]:


plt.bar(x=student['Hours'],height=student['Scores'])
plt.show()


# In[20]:


student


# In[21]:


numeric_columns=['Hours','Scores']


# In[26]:


sns.pairplot(student[numeric_columns])
plt.title("Pairplot")
plt.show()


# In[27]:


sns.boxplot(y=student['Scores'])
plt.title("Boxplot")
plt.show()


# In[28]:


student.isnull().sum()


# In[30]:


corre=student.corr()
top_corr_features=corre.index
plt.figure(figsize=(11,11))
g=sns.heatmap(student,annot=True,cmap='gist_rainbow',cbar_kws={"orientation":"vertical"},linewidths=1)


# In[31]:


sns.violinplot(y=student['Hours'])
plt.title("Violinplot")
plt.show()


# In[10]:



#create paiplot and two barplots
plt.figure(figsize=(16,6))
plt.subplot(131)
sns.pointplot(x="Hours", y="Scores", data=student)
plt.legend(['Hours = 1', 'Scores = 0'])
plt.show()


# In[16]:


x,y=student.loc[:,:'Hours'],student.loc[:,'Scores']


# In[17]:


x


# In[19]:


y


# In[20]:


x.shape


# In[21]:


y.shape


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[23]:


x=student.drop(['Scores'],axis=1)


# In[24]:


x


# In[25]:


x_train, x_test, y_train, y_test= train_test_split(x,y,random_state=10,test_size=0.3,shuffle=True)


# In[26]:


x_test


# In[29]:


print ("train_set_x shape: " + str(x_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(x_test.shape))
print ("test_set_y shape: " + str(y_test.shape))


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


linreg= LinearRegression()


# In[32]:


linreg.fit(x,y)


# In[34]:


from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor .fit(x_train, y_train)

print("completed")


# In[35]:


print(x_test)
y_pred = regressor.predict(x_test)


# In[42]:


plt.scatter(x,y)
plt.plot(x, y_pred, color='red')
plt.show()


# In[38]:


student = pd.DataFrame({'Actual' : y_test, 'Predicted': y_pred})
student


# In[39]:


Hours =9.25
own_pred = regressor.predict([[Hours]])
print("Number of hours = {}".format(Hours))
print("Predicted Score {}".format(own_pred[0]))


# In[ ]:




