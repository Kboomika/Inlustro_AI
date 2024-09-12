#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("Salary_Data.csv")
data
x=data[['YearsExperience']]
y=data['Salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
LinearRegression()
plt.scatter(x,y,color='blue')
plt.plot(x_test,y_pred,color='red')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Linear regression")
plt.show()


# In[ ]:




