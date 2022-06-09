#!/usr/bin/env python
# coding: utf-8

# # Data Visualization Project Group 3 
# ## Profiling Countries with Relation to Covid-19 Deaths 
# 
# 
# <b>Work developed by:</b><br> André Mendes | R2018 <br> Beatriz Serrador | R2018 <br>  Beatriz Silva | R20181173<br> Maria Cristina Jesus | R20181040<br> 
#  

# In[1]:


#Importing the libraries that we are going to use in this project
import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import Dash
import matplotlib.pyplot as plt
from matplotlib import ticker


# In[2]:


subPlots_Title_fontSize = 12
subPlots_xAxis_fontSize = 10
subPlots_yAxis_fontSize = 10 
subPlots_label_fontSize = 10
heatmaps_text_fontSize = 8

plots_Title_fontSize = 14
plots_Title_textColour = 'black'

plots_Legend_fontSize = 12
plots_Legend_textColour = 'black'


# In[5]:


covid_impact = pd.read_csv ('WHRData_2021.csv')
covid_impact.head()


# In[6]:


covid_impact.describe()


# In[7]:


#Check missing values
covid_impact.isna().sum()


# In[8]:


#Drop columns:
#All-cause death count 2017, 
#All-cause death count 2018, 
#All-cause death count 2019,
#All-cause death count 2020,
#Excess deaths in 2020 per 100,000 population, relative to 2017-2019 average.

covid_impact2= covid_impact.drop(['All-cause death count, 2017','All-cause death count, 2018','All-cause death count, 2019','All-cause death count, 2020','Excess deaths in 2020 per 100,000 population, relative to 2017-2019 average'] , axis=1)

#covid_impact2


# In[9]:


#Drop 2 rows: Somalian Region, North Cyprus
covid_impact2.drop(labels=[163,164])


# In[10]:


#Delete missing values

covid_impact2.dropna(inplace=True)
covid_impact2.isnull().sum()


# ### Variables Description

# In[11]:


# Columns datatypes
covid_impact2.info()


# In[12]:


plt.figure(figsize=(12,12))
sns.heatmap(covid_impact2.corr(), 
            vmin=-1, 
            vmax=1, 
            annot=True)

plt.title('correlation matrix of dataset')
plt.show()


# In[13]:


#Count values from Female head of government column
covid_impact2['Female head of government'].value_counts()


# In[14]:


# Percentage of Female Government Leaders

labels = 'Male', 'Female'
explode = (0, 0.1)  

fig1, ax1 = plt.subplots()
ax1.pie(covid_impact2['Female head of government'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()


# In[15]:


#Count values: Island
covid_impact2['Island'].value_counts()


# In[16]:


#Percentage of Country Islands

labels = 'Country Island', 'Not an Island'
explode = (0, 0.1) 

fig2, ax2 = plt.subplots()
ax2.pie(covid_impact2['Island'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.axis('equal')  

plt.show()


# In[17]:


#Scatter with 2 traces: Population 2019 and Population 2020
#ex2_trace1 = dict(type='scatter',
                  #x=ex2['Month'],
                  #y=ex2['2014 Sales'],
                  #name='2014 Sales'
                  #)

#ex2_trace2 = dict(type='scatter',
                  #x=ex2['Month'],
                  #y=ex2['2015 Sales'],
                  #name='2015 Sales'
                  #)

#ex2_data = [ex2_trace1, ex2_trace2]


#ex2_layout = dict(title=dict(text='Montly Sales over 2014 and 2015'),
                  #xaxis=dict(title='Months'),
                  #yaxis=dict(title='Sales in Monetatry Units')
                  #)

#ex2_fig = go.Figure(data=ex2_data, layout=ex2_layout)

#ex2_fig.show()




scatter_Pop19 = dict(type='bar',
                  x=covid_impact2['Continent'],
                  y=covid_impact2['Population 2019'],
                  name='2019 Pop'
                  )


scatter_Pop20 = dict(type='bar',
                  x=covid_impact2['Continent'],
                  y=covid_impact2['Population 2020'],
                  name='2020 Pop'
                  )

scatter_data = [scatter_Pop19, scatter_Pop20]

scatter_layout = dict(title=dict(text='Population per Continent: 2019|2020'),
                  xaxis=dict(title='Continents'),
                  yaxis=dict(title='Population')
                  )

scatter_fig = go.Figure(data=scatter_data, layout=scatter_layout)

scatter_fig.show()






# In[18]:


#TOP 10 Countries: Population 2020

data_10countriespop = covid_impact2.loc[:,['Country name','Population 2020']]

datasort_10countriespop = data_10countriespop.sort_values(ascending=False, by='Population 2020').head(10)

fig_10countriespop = px.bar(datasort_10countriespop, 
                         x='Country name', 
                         y='Population 2020', 
                         )
                         
fig_10countriespop.update_layout(title_text='Top10 Countries: Population 2020')
fig_10countriespop.show()


# In[19]:


#Pie-Chart: deaths per continent


pie_chart_labels = covid_impact2['Continent']
pie_chart_values = covid_impact2 ['COVID-19 deaths per 100,000 population in 2020']

pie_chart_data = dict(type='pie',
                    labels=pie_chart_labels,
                    values=pie_chart_values,
                    name='Pie Continent'
                    )

pie_chart_layout = dict (title=dict(text='Covid-19 Deaths per Continent')
                  )
pie_chart_fig = go.Figure(data=pie_chart_data, layout=pie_chart_layout)

pie_chart_fig.show()


# In[20]:


#TOP 10 Countries: Covid-19 Deaths

data_10countries = covid_impact2.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020']]

datasort_10countries = data_10countries.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020').head(10)

fig_10countries = px.bar(datasort_10countries, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020', 
                         )
                         
fig_10countries.update_layout(title_text='Top10 Countries: Covid-19 Deaths')
fig_10countries.show()


# In[21]:


# Bubble-Chart: Median age vs Covid-19 Deaths

figure_age_deaths = px.scatter(covid_impact2, x='Median age', y='COVID-19 deaths per 100,000 population in 2020', size='Population 2020', color='Continent', size_max=100, hover_name='Country name', log_x=True
                )
figure_age_deaths.update_layout(title_text='Median age VS Covid-19 Deaths per 100,000')
figure_age_deaths.show()


# ### Data Analysis by Continents

# In[22]:


#Count values: Continent
covid_impact2['Continent'].value_counts()


# #### Europe

# In[23]:


#Europe Data

covid_impactEurope = covid_impact2[(covid_impact2['Continent'] == 'Europe')]
#covid_impactEurope


# In[24]:


#Covid-19 Deaths per European Country

#x_covidimpactEurope = covid_impactEurope['Country name']
#y_covidimpactEurope = covid_impactEurope['COVID-19 deaths per 100,000 population in 2020']


#data_europe = dict(type='bar', x=x_covidimpactEurope, y=y_covidimpactEurope)

#layout_europe = dict(title=dict(text='Europe Data'),
               #yaxis=dict(title='COVID-19 deaths per 100,000 population in 2020'))



#fig_europe = go.Figure(data=data_europe, layout=layout_europe)

#fig_europe.update_xaxes(categoryorder='total descending')

#fig_europe.show()

fig_europe1 = px.bar(covid_impactEurope, x='Country name', y='COVID-19 deaths per 100,000 population in 2020',
            hover_data=['Median age'], color='Median age',
            labels={'COVID-19 deaths per 100,000 population in 2020':'COVID-19 deaths per 100,000 population in 2020'}, height=600, title= "COVID-19 deaths VS Median Age", 
            color_continuous_scale='magenta')
fig_europe1.show()


# In[25]:


# Index of Exposure of Covid-19 infections and Index of Institutional Trust: Bar Chart with Colored bar

fig_index = px.bar(covid_impactEurope, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Index of institutional trust'], color='Index of institutional trust',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index of Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs Index of Instituional Trust",
            color_continuous_scale='sunset')
fig_index.show()


# In[26]:


# Log of average distance to SARS countries 


# #### Africa

# In[27]:


# Africa Data
covid_impactAfrica = covid_impact2[(covid_impact2['Continent'] == 'Africa')]
#covid_impactAfrica


# In[28]:


# Africa Covid-19 Related Deaths

x_covidimpactAfrica = covid_impactAfrica['Country name']
y_covidimpactAfrica = covid_impactAfrica['COVID-19 deaths per 100,000 population in 2020']

data_africa = dict(type='bar', x=x_covidimpactAfrica, y=y_covidimpactAfrica)

layout_africa = dict(title=dict(text='Africa Covid-19 Related Deaths'),
               yaxis=dict(title='COVID-19 deaths per 100,000 population in 2020'))



fig_africa = go.Figure(data=data_africa, layout=layout_africa)

fig_africa.update_xaxes(categoryorder='total descending')

fig_africa.show()


# In[29]:


# Index Covid-19 Exposure in the African Continent

fig_africa = px.bar(covid_impactAfrica, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure in the African Continent",
            color_continuous_scale='tealgrn'
            )
fig_africa.show()


# #### Asia

# In[30]:


#Asia Data
covid_impactAsia = covid_impact2[(covid_impact2['Continent'] == 'Asia')]
covid_impactAsia


# In[31]:


x_covidimpactAsia = covid_impactAsia['Country name']
y_covidimpactAsia = covid_impactAsia['COVID-19 deaths per 100,000 population in 2020']

data_asia = dict(type='bar', x=x_covidimpactAsia, y=y_covidimpactAsia)

layout_asia = dict(title=dict(text='Covid-19 Deaths in the Asian Continent'),
               yaxis=dict(title='COVID-19 deaths per 100,000 population in 2020'))



fig_asia = go.Figure(data=data_asia, layout=layout_asia)

fig_asia.update_xaxes(categoryorder='total descending')

fig_asia.show()


# In[32]:


# Bubble-Chart: Median age vs Covid-19 Deaths

#figure_age_deaths = px.scatter(covid_impact2, x='Median age', y='COVID-19 deaths per 100,000 population in 2020', size='Population 2020', color='Continent', size_max=100, hover_name='Country name', log_x=True
                #)
#figure_age_deaths.update_layout(title_text='Median age VS Covid-19 Deaths per 100,000')
#figure_age_deaths.show()


#Bubble-Chart: Index Covid-19 Exposure VS Income GINI Coefficient

figure_index_GINI_Asia = px.scatter(covid_impactAsia, x='COVID-19 deaths per 100,000 population in 2020', y='Gini coefficient of income', size='Population 2020', color='Country name', size_max=100, hover_name='Country name', log_x=True
                )
figure_index_GINI_Asia.update_layout(title_text='Income GINI Coefficient VS Covid-19 Deaths')
figure_index_GINI_Asia.show()




# #### North-America

# In[33]:


#North America Data
covid_impactNorthAmerica = covid_impact2[(covid_impact2['Continent'] == 'North America')]
covid_impactNorthAmerica


# In[34]:


#x_covidimpactNorthAmerica = covid_impactNorthAmerica['Country name']
#y_covidimpactNorthAmerica = covid_impactNorthAmerica['Index of exposure to COVID-19  infections in other countries as of March 31']

#data_northamerica = dict(type='bar', x=x_covidimpactNorthAmerica, y=y_covidimpactNorthAmerica)

#layout_northamerica = dict(title=dict(text='Covid-19 Index Exposure in North America'),
               #yaxis=dict(title='Index Exposure'))



#fig_northamerica = go.Figure(data=data_northamerica, layout=layout_northamerica)

#fig_northamerica.update_xaxes(categoryorder='total descending')

#fig_northamerica.show()




fig_northamerica = px.bar(covid_impactNorthAmerica, x='Country name', y='COVID-19 deaths per 100,000 population in 2020',
            hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
            labels={'COVID-19 deaths per 100,000 population in 2020':'COVID-19 deaths per 100,000 population in 2020'}, height=600, title= "COVID-19 deaths in North America", 
            color_continuous_scale='ylgn')
fig_northamerica.show()


# #### South-America

# In[35]:


#South-America Data
covid_impactSouthAmerica = covid_impact2[(covid_impact2['Continent'] == 'South America')]
covid_impactSouthAmerica


# In[36]:


#Covid-19 Deaths: South America

x_covidimpactSouthAmerica = covid_impactSouthAmerica['Country name']
y_covidimpactSouthAmerica = covid_impactSouthAmerica['COVID-19 deaths per 100,000 population in 2020']

data_southamerica = dict(type='bar', x=x_covidimpactSouthAmerica, y=y_covidimpactSouthAmerica)

layout_southamerica = dict(title=dict(text='South-America Covid-19 Deaths'),
               yaxis=dict(title='COVID-19 deaths per 100,000 population in 2020'))



fig_southamerica = go.Figure(data=data_southamerica, layout=layout_southamerica)

fig_southamerica.update_xaxes(categoryorder='total descending')

fig_southamerica.show()


# #### Oceania

# In[37]:


#Oceania Data
covid_impactOceania = covid_impact2[(covid_impact2['Continent'] == 'Oceania')]
covid_impactOceania


# In[38]:


#x_covidimpactOceania = covid_impactOceania['Country name']
#y_covidimpactOceania = covid_impactOceania['COVID-19 deaths per 100,000 population in 2020']

#data_oceania = dict(type='bar', x=x_covidimpactOceania, y=y_covidimpactOceania)

#layout_oceania = dict(title=dict(text='South America Data'),
               #yaxis=dict(title='Covid-19 deaths'))



#fig_oceania = go.Figure(data=data_oceania, layout=layout_oceania)

#fig_oceania.update_xaxes(categoryorder='total descending')

#fig_oceania.show()


#Pie-chart: Oceania Covid-19 Deaths


pie_chart_labels_oce = covid_impactOceania['Country name']
pie_chart_values_oce = covid_impactOceania['COVID-19 deaths per 100,000 population in 2020']

pie_chart_data_oce = dict(type='pie',
                    labels=pie_chart_labels_oce,
                    values=pie_chart_values_oce,
                    name='Oceania Pie-Charts'
                    )

pie_chart_layout_oce = dict (title=dict(text='Oceania Covid-19 Deaths')
                  )
pie_chart_fig_oce = go.Figure(data=pie_chart_data_oce, layout=pie_chart_layout_oce)

pie_chart_fig_oce.show()


# In[39]:


#teste


fig_teste = px.scatter(covid_impact2[(covid_impact2['Continent'] == 'Europe')], 
                 x='Median age', 
                 y='Gini coefficient of income', 
                 color='Country name',
                 trendline='ols',
                 trendline_scope='overall',
                 trendline_color_override='black'
                )
fig_teste.show()


# In[40]:


data = covid_impact2.loc[:,['Country name','Population 2020','Population 2019']].sort_values(by='Population 2019',
                                                                                   ascending=False).head(10)

# plotting go figure for grouped bar chart

fig_population = go.Figure(data=[go.Bar(name='Population 2019',x=data['Country name'],y=data['Population 2019']),
                      go.Bar(name='Population 2020',x=data['Country name'],y=data['Population 2020'])
                     ])

fig_population.update_layout(barmode='group', title_text='Top10 countries with most population')
fig_population.show()


# In[41]:


#App itself

app = dash.Dash(__name__)

server = app.server

#test
app.layout = html.Div(children=[
    html.H1(children='PROFILING COUNTRIES WITH RELATION TO COVID-19 DEATHS'),

    html.Div(children='''
        Example of html Container
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig_teste
    ) ,
    dcc.Checklist(covid_impact2.Continent.unique(), covid_impact2.Continent.unique()[0:5])
])

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




