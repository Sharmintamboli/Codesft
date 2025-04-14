#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv(r'Desktop\Iris.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[11]:


X = df.drop('species', axis=1)
y = df['species']


# In[13]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=42)


# In[19]:


#train model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)


# In[21]:


y_predict=model.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict, target_names=label_encoder.classes_))


# In[ ]:




