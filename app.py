#!/usr/bin/env python
# coding: utf-8

# # Data Visualization Project Group 3 
# ## Profiling Countries with Relation to Covid-19 Deaths 
# 
# 
# <b>Work developed by:</b><br> André Mendes | R2018 <br> Beatriz Serrador | R2018 <br>  Beatriz Silva | R20181173<br> Maria Cristina Jesus | R20181040<br> 
#  

# In[99]:


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


# In[116]:


covid_impact = pd.read_csv (r'WHRData_2021.csv',sep = ",")
covid_impact.head()

#r'WHRData_2021.csv'


# In[103]:


covid_impact.describe()


# In[104]:


#Check missing values
covid_impact.isna().sum()


# In[105]:


#Drop columns:
#All-cause death count 2017, 
#All-cause death count 2018, 
#All-cause death count 2019,
#All-cause death count 2020,
#Excess deaths in 2020 per 100,000 population, relative to 2017-2019 average.

covid_impact2= covid_impact.drop(['All-cause death count, 2017','All-cause death count, 2018','All-cause death count, 2019','All-cause death count, 2020','Excess deaths in 2020 per 100,000 population, relative to 2017-2019 average'] , axis=1)

covid_impact2


# In[106]:


#Drop 2 rows: Somalian Region, North Cyprus
covid_impact2.drop(labels=[163,164])


# In[107]:


#Delete missing values

covid_impact2.dropna(inplace=True)
covid_impact2.isnull().sum()


# ### Variables Description

# In[108]:


# Columns datatypes
covid_impact2.info()


# In[109]:


#Count values from Female head of government column
covid_impact2['Female head of government'].value_counts()


# In[110]:


# TENHO QUE MUDAR ESTE PIE CHART PARA O CÓDIGO DO PROF

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Female', 'Male'
sizes = [23, 140]
explode = (0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[142]:


#tentei

labels_1 = covid_impact2['Female head of government']

#Sum of all Emissions for each continent
values_1 = covid_impact2['Female head of government']

data_8 = dict(type='pie', labels=labels_1, values=values_1)

layout_8 = dict(title=dict(text='Ratio of females as the head of government'))

fig_8 = go.Figure(data=[data_8], layout=layout_8)

fig_8.show()


# In[143]:


#Count values: Island
covid_impact2['Island'].value_counts()


# In[144]:


#VISUALIZAÇÃO PARA COUNTRY ISLAND
covid_impactIsland = covid_impact2[(covid_impact2['Island'] == 1)]
covid_impactIsland


# In[ ]:





# In[112]:


#Count values: Continent
covid_impact2['Continent'].value_counts()


# In[145]:


#Bar-chart: Continents
#x_bar = df_europe_2000['country_name']
#y_bar = df_europe_2000['CO2_emissions']

#data_4 = dict(type='bar', x=x_bar, y=y_bar)

#layout_4 = dict(title=dict(text='Europe Emissions in the year 2000'),
               #yaxis=dict(title='Emissions'))



#layout_4_1 = dict(title=dict(text='Europe Emissions in the year 2000'),
                  #yaxis=dict(type='log',
                            #title='Emissions Log Scaled'))
        

        
x_bar = covid_impact2['Continent']
y_bar = covid_impact2['Population 2020']

data_continents = dict(type='bar', x=x_bar, y=y_bar)

layout_continents = dict(title=dict(text='Population 2020'),
               yaxis=dict(title='Population 2020'))

fig_continents = go.Figure(data=data_continents, layout=layout_continents)

fig_continents.show()


# ### Continents Data

# In[146]:


#Europe Data
covid_impactEurope = covid_impact2[(covid_impact2['Continent'] == 'Europe')]
covid_impactEurope


# In[160]:


x_covidimpactEurope = covid_impactEurope['Country name']
y_covidimpactEurope = covid_impactEurope['COVID-19 deaths per 100,000 population in 2020']


data_europe = dict(type='bar', x=x_covidimpactEurope, y=y_covidimpactEurope)

layout_europe = dict(title=dict(text='Europe Data'),
               yaxis=dict(title='Covid-19 deaths'))



fig_europe = go.Figure(data=data_europe, layout=layout_europe)

fig_europe.update_xaxes(categoryorder='total descending')

fig_europe.show()


# In[162]:


#Africa Data
covid_impactAfrica = covid_impact2[(covid_impact2['Continent'] == 'Africa')]
covid_impactAfrica


# In[165]:


x_covidimpactAfrica = covid_impactAfrica['Country name']
y_covidimpactAfrica = covid_impactAfrica['COVID-19 deaths per 100,000 population in 2020']

data_africa = dict(type='bar', x=x_covidimpactAfrica, y=y_covidimpactAfrica)

layout_africa = dict(title=dict(text='Africa Data'),
               yaxis=dict(title='Covid-19 deaths'))



fig_africa = go.Figure(data=data_africa, layout=layout_africa)

fig_africa.update_xaxes(categoryorder='total descending')

fig_africa.show()


# In[166]:


#Asia Data
covid_impactAsia = covid_impact2[(covid_impact2['Continent'] == 'Asia')]
covid_impactAsia


# In[167]:


x_covidimpactAsia = covid_impactAsia['Country name']
y_covidimpactAsia = covid_impactAsia['COVID-19 deaths per 100,000 population in 2020']

data_asia = dict(type='bar', x=x_covidimpactAsia, y=y_covidimpactAsia)

layout_asia = dict(title=dict(text='Asia Data'),
               yaxis=dict(title='Covid-19 deaths'))



fig_asia = go.Figure(data=data_asia, layout=layout_asia)

fig_asia.update_xaxes(categoryorder='total descending')

fig_asia.show()


# In[168]:


#North America Data
covid_impactNorthAmerica = covid_impact2[(covid_impact2['Continent'] == 'North America')]
covid_impactNorthAmerica


# In[169]:


x_covidimpactNorthAmerica = covid_impactNorthAmerica['Country name']
y_covidimpactNorthAmerica = covid_impactNorthAmerica['COVID-19 deaths per 100,000 population in 2020']

data_northamerica = dict(type='bar', x=x_covidimpactNorthAmerica, y=y_covidimpactNorthAmerica)

layout_northamerica = dict(title=dict(text='North America Data'),
               yaxis=dict(title='Covid-19 deaths'))



fig_northamerica = go.Figure(data=data_northamerica, layout=layout_northamerica)

fig_northamerica.update_xaxes(categoryorder='total descending')

fig_northamerica.show()


# In[170]:


#North America Data
covid_impactSouthAmerica = covid_impact2[(covid_impact2['Continent'] == 'South America')]
covid_impactSouthAmerica


# In[171]:


x_covidimpactSouthAmerica = covid_impactSouthAmerica['Country name']
y_covidimpactSouthAmerica = covid_impactSouthAmerica['COVID-19 deaths per 100,000 population in 2020']

data_southamerica = dict(type='bar', x=x_covidimpactSouthAmerica, y=y_covidimpactSouthAmerica)

layout_southamerica = dict(title=dict(text='South America Data'),
               yaxis=dict(title='Covid-19 deaths'))



fig_southamerica = go.Figure(data=data_southamerica, layout=layout_southamerica)

fig_southamerica.update_xaxes(categoryorder='total descending')

fig_southamerica.show()


# In[172]:


#Oceania Data
covid_impactOceania = covid_impact2[(covid_impact2['Continent'] == 'Oceania')]
covid_impactOceania


# In[174]:


x_covidimpactOceania = covid_impactOceania['Country name']
y_covidimpactOceania = covid_impactOceania['COVID-19 deaths per 100,000 population in 2020']

data_oceania = dict(type='bar', x=x_covidimpactOceania, y=y_covidimpactOceania)

layout_oceania = dict(title=dict(text='South America Data'),
               yaxis=dict(title='Covid-19 deaths'))



fig_oceania = go.Figure(data=data_oceania, layout=layout_oceania)

fig_oceania.update_xaxes(categoryorder='total descending')

fig_oceania.show()


# In[ ]:


#data_3 = dict(type='scatter', x=x_portugal, y=y_portugal)

#layout_3 = dict(title=dict(
                    #text='Portugal Emissions from 1990 until 2015'
                    #),
             #xaxis=dict(title='Years'),
             #yaxis=dict(title='CO2 Emissions'))


#fig_3 = go.Figure(data=[data_3], layout=layout_3)

#fig_3.show(renderer='png')

#data_europe = dict(type='scatter', x=x_covid_impactEurope , y=y_covid_impactEurope)

#layout_europe= dict(title=dict(
                    #text='Europe Data'
                    #),
             #xaxis=dict(title='Country name'),
             #yaxis=dict(title='COVID-19 deaths per 100,000 population in 2020'))


#data_europe = go.Figure(data=[data_europe], layout=layout_europe)

#data_europe.show(renderer='png')




# In[178]:


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


# In[176]:


data = covid_impact2.loc[:,['Country name','Population 2020','Population 2019']].sort_values(by='Population 2019',
                                                                                   ascending=False).head(10)

# plotting go figure for grouped bar chart

fig_population = go.Figure(data=[go.Bar(name='Population 2019',x=data['Country name'],y=data['Population 2019']),
                      go.Bar(name='Population 2020',x=data['Country name'],y=data['Population 2020'])
                     ])

fig_population.update_layout(barmode='group', title_text='Top10 countries with most population')
fig_population.show()


# In[175]:


plt.figure(figsize=(12,12))
sns.heatmap(covid_impact2.corr(), 
            vmin=-1, 
            vmax=1, 
            annot=True)

plt.title('correlation matrix of dataset')
plt.show()


# In[88]:


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




