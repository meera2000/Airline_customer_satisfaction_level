#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# In[2]:


cd D:\FT\python\Ml\Assignment


# Part-1: Data Exploration and Pre-processing

# In[3]:


# 1) load the given dataset 
data=pd.read_csv("Python_Project_5_Dec.csv")


# In[4]:


# 2) print all the column names 
data.columns


# In[5]:


# 3) describe the data 
data.describe()


# In[6]:


# 4) Drop the column ‘Unnamed’
data.drop("Unnamed: 0",axis=1,inplace=True)


# In[7]:


# 5) Replace all the “ “ in column with “_” 
data.columns=[col.replace(" ","_") for col in data.columns]


# In[9]:


data.columns


# In[10]:


# 6) Plot the number of satisfied customers and the number of unsatisfied customers 
# df["satisfaction"]=[1 if each=="satisfied" else 0 for each in df.satisfaction]
data['satisfaction']=[1 if val=='satisfied' else 0 for val in data.satisfaction]
data['satisfaction'].value_counts()


# In[11]:


# plt.bar(data['satisfaction'].value_counts())


# In[12]:


# 7) Plot the mean value of satisfaction of male and female customers 
data[["Gender","satisfaction"]].groupby(["Gender"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)


# In[13]:


# 8) Plot the mean value of satisfaction of customers with respect to Age.
data[['Age','satisfaction']].groupby(["Age"],as_index=False).mean().sort_values(by="satisfaction",ascending=False)


# In[14]:


# 9) Plot the mean value of satisfaction of customers with respect to Food_and_drink.
data[['Food_and_drink','satisfaction']].groupby(['Food_and_drink'],as_index=False).mean().sort_values(by="satisfaction",ascending=False)


# In[15]:


# 10) Display a boxplot for Flight_Distance 
plt.boxplot(data['Flight_Distance'])
plt.show()


# In[89]:


# 11) Display a boxplot for Checkin_service
plt.boxplot(data['Checkin_service'])
plt.show()


# In[100]:


#12) Find all the Null values 
data.isnull().sum()


# In[16]:


# 13) Drop all the na values 
data.dropna(inplace=True)


# In[17]:


# 14) Find the unique values in Flight_Distance
data['Flight_Distance'].unique()


# Part-2: Working with Models

# In[18]:


# 1) Perform encoding in columns Gender, Customer_Type,Type_of_Travel, and Class.
from sklearn.preprocessing import LabelEncoder


# In[19]:


label_encoder=LabelEncoder()


# In[20]:


data['Gender']=label_encoder.fit_transform(data['Gender'])
data['Customer_Type']=label_encoder.fit_transform(data['Customer_Type'])
data['Type_of_Travel']=label_encoder.fit_transform(data['Type_of_Travel'])
data['Class']=label_encoder.fit_transform(data['Class'])


# In[21]:


# 2) Drop the column id
data.drop('id',axis=1,inplace=True)


# In[22]:


# 3) Create the features and target Data 
from sklearn.model_selection import train_test_split


# In[23]:


x=data.drop('satisfaction',axis=1)
y=data.satisfaction


# In[24]:


print(x)
print(y)


# In[25]:


# 4) Perform scaling on features data 
from sklearn.preprocessing import StandardScaler


# In[26]:


sd=StandardScaler()
x=sd.fit_transform(x)


# In[27]:


# 6) Split the data in training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[108]:


# 5) Fit the decision tree model with various parameters
model=DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[109]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score


# In[111]:


# 7) Create a function to display precision score, recall score, accuracy, 
# classification report, confusion matrix, F1 Score
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("pricison_score: ",precision_score(y_test, y_pred))
print("recall_score: ",recall_score(y_test, y_pred))
print("Accuracy = {}".format(accuracy))
print(classification_report(y_test,y_pred,digits=5))
print(confusion_matrix(y_test,y_pred))

