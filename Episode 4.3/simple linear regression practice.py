#!/usr/bin/env python
# coding: utf-8

# In[5]:



# use for data manipulation
import pandas
# use for plot
import matplotlib.pyplot as plt 
# mathemetical operation
import numpy as np

weather_data = pandas.read_csv("C:/Users/msin2/Documents/college/spring_2021/stat-212/TheDataSeries/WeatherData.csv")
weather_data


# In[37]:


x = weather_data["Temperature(C)"]
y = weather_data["Humidity"]
plt.scatter(x, y)
plt.xlabel("Temperature(C)")
plt.ylabel("Humidity")


# In[30]:


from sklearn.linear_model import LinearRegression
x = weather_data["Temperature(C)"].values.reshape(-1, 1)
y = weather_data["Humidity"].values.reshape(-1, 1) 

lr_model = LinearRegression()
lr_model.fit(x, y)
y_pred = lr_model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel("Temperature(C)")
plt.ylabel("Humidity")


# In[32]:


theta_0 = lr_model.intercept_
theta_1 = lr_model.coef_
theta_0, theta_1


# In[36]:


y_pred = lr_model.predict(np.array([10]).reshape(1, 1))
y_pred


# In[ ]:




