#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install jupyter-dash


# In[72]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html


# In[73]:


covid_impact = pd.read_csv (r'C:\Users\andre\Downloads\archive\WHRData2021.csv',sep = ",")
covid_impact.head()


# In[74]:


covid_impact.describe()


# In[75]:


covid_impact.isna().sum()


# In[76]:


covid_impact2= covid_impact.drop(['All-cause death count, 2017','All-cause death count, 2018','All-cause death count, 2019','All-cause death count, 2020','Excess deaths in 2020 per 100,000 population, relative to 2017-2019 average'] , axis=1)

covid_impact2.head()


# In[77]:


covid_impact2.dropna(inplace=True)

covid_impact2.isnull().sum()


# In[78]:


# dar a descobrir algumas variáveis


# In[79]:


#teste


fig = px.scatter(covid_impact2, 
                 x='Median age', 
                 y='Gini coefficient of income', 
                 color='Country name',
                 trendline='ols',
                 trendline_scope='overall',
                 trendline_color_override='black'
                )
fig.show()


# In[80]:


data = covid_impact2.loc[:,['Country name','Population 2020','Population 2019']].sort_values(by='Population 2019',
                                                                                   ascending=False).head(10)

# plotting go figure for grouped bar chart

fig = go.Figure(data=[go.Bar(name='Population 2019',x=data['Country name'],y=data['Population 2019']),
                      go.Bar(name='Population 2020',x=data['Country name'],y=data['Population 2020'])
                     ])

fig.update_layout(barmode='group', title_text='Top10 countries with most population')
fig.show()


# In[81]:


data = covid_impact2.loc[:,['Country name','Population 2020','COVID-19 deaths per 100,000 population in 2020']].sort_values(by='COVID-19 deaths per 100,000 population in 2020',
                                                                                   ascending=False).head(10)

# plotting go figure for grouped bar chart

fig = go.Figure(data=[go.Bar(name='COVID-19 deaths per 100,000 population in 2020',x=data['Country name'],y=data['COVID-19 deaths per 100,000 population in 2020']),
                      go.Bar(name='Population 2020',x=data['Country name'],y=data['Population 2020'])
                     ])

fig.update_layout(barmode='group', title_text='teste')
fig.show()


# In[82]:


covid_impact2.info()


# In[83]:


covid_impact2['Female head of government'].value_counts()


# In[84]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Female', 'Male'
sizes = [23, 140]
explode = (0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[70]:


plt.figure(figsize=(12,12))
sns.heatmap(covid_impact2.corr(), 
            vmin=-1, 
            vmax=1, 
            annot=True)

plt.title('correlation matrix of dataset')
plt.show()


# In[ ]:




