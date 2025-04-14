#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_csv(r'Desktop\\advertising.csv')


# In[6]:


df.info()


# In[7]:


df.head()


# In[8]:


df.tail()


# In[10]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


#preprocessing and modelling training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# In[14]:


X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']                       # Target


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[19]:


y_pred = model.predict(X_test)


# In[20]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²) Score:", r2)


# In[21]:


#Show predicted vs actual sales
print("\nPredicted Sales:")
print(y_pred)

print("\nActual Sales:")
print(y_test.values)


# In[ ]:




