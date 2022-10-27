#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go 


# In[26]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/tips.csv')


# In[27]:


data.head()


# # Features explanation
# 1.total_bill: Total bill in dollars including tax
# 2.tip: Tip given to waiter in dollars
# 3.sex: gender of the person paying the bill
# 4.smoker: whether the person smoked or not
# 5.day: day of the week
# 6.time: lunch or dinner
# 7.size: number of people

# In[28]:


# Tips vs bills
figure=px.scatter(data,x='total_bill',y='tip',size='size', color='day', trendline='ols')
figure.show()


# In[29]:


figure=px.scatter(data,x='total_bill',y='tip',size='size', color='sex', trendline='ols')
figure.show()


# In[30]:


figure=px.scatter(data,x='total_bill',y='tip',size='size', color='time', trendline='ols')
figure.show()


# In[31]:


# Now lets find out the days which tips are give most
figure=px.pie(data, values='tip',names='day',hole=0.5)
figure.show()


# ## According to the visualization saturday is the most tips are given to the waiters and sunday comes next.

# In[32]:


figure=px.pie(data, values='tip',names='sex',hole=0.5)
figure.show()


# ## Most tips are given by male 

# In[33]:


figure=px.pie(data, values='tip',names='smoker',hole=0.5)
figure.show()


# ## Non-smoker gives tips more

# In[34]:


figure=px.pie(data, values='tip',names='time',hole=0.5)
figure.show()


# ## So from the above visualization waiter is tipped more in the dinner 

# In[35]:


# Waiter tips prediction model
data['sex']=data['sex'].map({'Female':0,'Male':1})
data['smoker']=data['smoker'].map({'No':0,'Yes':1})
data['day']=data['day'].map({'Thur':0,'Fri':1,'Sat':2,'Sun':3})
data['time']=data['time'].map({'Lunch':0,'Dinner':1})


# In[36]:


data.head()


# In[40]:


x=np.array(data[['total_bill','sex','smoker','day','time','size']])
y=np.array(data['tip'])
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=42)


# In[41]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)


# In[42]:


# features = [[total_bill, "sex", "smoker", "day", "time", "size"]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])
model.predict(features)


# In[ ]:





# In[ ]:




