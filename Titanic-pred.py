#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd


# In[13]:


import numpy as np


# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


df=pd.read_csv(r'Desktop//Titanic-DAtaset.csv')


# In[24]:


df.head()


# In[25]:


df.tail()


# In[29]:


df.describe()


# In[35]:


#Handling the missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[36]:


df


# In[37]:


#Drop rows with remaining null values
df.dropna(inplace=True)


# In[53]:


df.isnull().sum()


# In[43]:


from sklearn.preprocessing import LabelEncoder
# Create a dictionary to store encoders (for possible inverse_transform later)
label_encoders = {}

# These are the columns that contain categorical data (text values)
categorical_cols = ['Sex', 'Embarked']
for col in categorical_cols:
    le = LabelEncoder()                      # Create a new LabelEncoder for each column
    df[col] = le.fit_transform(df[col])      # Fit the encoder and transform the data to numeric
    label_encoders[col] = le                 # Save the encoder if you ever need to reverse it


# In[45]:


# Select relevant features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']


# In[46]:


from sklearn.model_selection import train_test_split
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[47]:


from sklearn.linear_model import LogisticRegression
# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# In[62]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#  Evaluation
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[66]:


#Plotting feature importance (coefficients)
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define your custom class labels
labels = ['Not Survived', 'Survived']

# Create a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# In[ ]:




