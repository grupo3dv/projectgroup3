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
import dash_bootstrap_components as dbc
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

#Pie-Chart:Female Leaders by Continent

pie_chart_labelsf = covid_impact2['Continent']
pie_chart_valuesf = covid_impact2['Female head of government']

pie_chart_dataf = dict(type='pie',
                    labels=pie_chart_labelsf,
                    values=pie_chart_valuesf,
                    name='Pie Continent'
                    )

pie_chart_layoutf = dict (title=dict(text='Percentage of Female Leaders per Continent')
                  )
pie_chart_female = go.Figure(data=pie_chart_dataf, layout=pie_chart_layoutf)

pie_chart_female.show()
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

data_10countries = covid_impact2.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries = data_10countries.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020')

fig_10countries = px.bar(datasort_10countries, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries.update_layout(title_text='Top10 Countries: Covid-19 Deaths')
fig_10countries.show()

# In[21]:


# Bubble-Chart: Median age vs Covid-19 Deaths

bubble_age_deaths = px.scatter(covid_impact2, x='Median age', y='COVID-19 deaths per 100,000 population in 2020', size='Population 2020', color='Continent', size_max=100, hover_name='Country name', log_x=True
                )
bubble_age_deaths.update_layout(title_text='Median age VS Covid-19 Deaths per 100,000')
bubble_age_deaths.show()


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

fig_europe1 = px.bar(covid_impactEurope, x='Country name', y='COVID-19 deaths per 100,000 population in 2020',
            hover_data=['Median age'], color='Median age',
            labels={'COVID-19 deaths per 100,000 population in 2020':'COVID-19 deaths per 100,000 population in 2020'}, height=600, title= "COVID-19 deaths VS Median Age", 
            color_continuous_scale='magenta')
fig_europe1.show()


# In[25]:
# Index of Exposure of Covid-19 infections and Index of Institutional Trust in Europe: Bar Chart with Colored bar

europe_exposure_trust = px.bar(covid_impactEurope, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Index of institutional trust'], color='Index of institutional trust',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index of Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs Index of Instituional Trust in Europe",
            color_continuous_scale='sunset')
europe_exposure_trust.show()


# In[26]:

# Index Covid-19 Exposure in the European Continent

europe_exposure_income = px.bar(covid_impactEurope, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs GINI Coefficient of Income in Europe",
            color_continuous_scale='tealgrn'
            )
europe_exposure_income.show()


# #### Africa

# In[27]:


# Africa Data
covid_impactAfrica = covid_impact2[(covid_impact2['Continent'] == 'Africa')]
#covid_impactAfrica


# In[28]:

# Index of Exposure of Covid-19 infections and Index of Institutional Trust in Africa: Bar Chart with Colored bar

africa_exposure_trust = px.bar(covid_impactAfrica, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Index of institutional trust'], color='Index of institutional trust',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index of Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs Index of Instituional Trust in Africa",
            color_continuous_scale='sunset')
africa_exposure_trust.show()



# In[29]:


# Index Covid-19 Exposure in the African Continent

africa_exposure_income = px.bar(covid_impactAfrica, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs GINI Coefficient of Income in Africa",
            color_continuous_scale='tealgrn'
            )
africa_exposure_income.show()



# #### Asia

# In[30]:


#Asia Data
covid_impactAsia = covid_impact2[(covid_impact2['Continent'] == 'Asia')]
covid_impactAsia


# In[31]:


# Index of Exposure of Covid-19 infections and Index of Institutional Trust in Asia: Bar Chart with Colored bar

asia_exposure_trust = px.bar(covid_impactAsia, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Index of institutional trust'], color='Index of institutional trust',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index of Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs Index of Instituional Trust in Asia",
            color_continuous_scale='sunset')
asia_exposure_trust.show()

# Index Covid-19 Exposure in the Asian Continent

asia_exposure_income = px.bar(covid_impactAsia, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs GINI Coefficient of Income in Asia",
            color_continuous_scale='tealgrn'
            )
asia_exposure_income.show()



# #### North-America

# In[33]:


#North America Data
covid_impactNorthAmerica = covid_impact2[(covid_impact2['Continent'] == 'North America')]
covid_impactNorthAmerica


# In[34]:


# Index of Exposure of Covid-19 infections and Index of Institutional Trust in North America: Bar Chart with Colored bar

northamerica_exposure_trust = px.bar(covid_impactNorthAmerica, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Index of institutional trust'], color='Index of institutional trust',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index of Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs Index of Instituional Trust in North America",
            color_continuous_scale='sunset')
northamerica_exposure_trust.show()

# Index Covid-19 Exposure in the North American Continent

northamerica_exposure_income = px.bar(covid_impactNorthAmerica, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs GINI Coefficient of Income in North America",
            color_continuous_scale='tealgrn'
            )
northamerica_exposure_income.show()

# #### South-America

# In[35]:


#South-America Data
covid_impactSouthAmerica = covid_impact2[(covid_impact2['Continent'] == 'South America')]
covid_impactSouthAmerica


# In[36]:


# Index of Exposure of Covid-19 infections and Index of Institutional Trust in South America: Bar Chart with Colored bar

southamerica_exposure_trust = px.bar(covid_impactSouthAmerica, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Index of institutional trust'], color='Index of institutional trust',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index of Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs Index of Instituional Trust in South America",
            color_continuous_scale='sunset')
southamerica_exposure_trust.show()

# Index Covid-19 Exposure in the South American Continent

southamerica_exposure_income = px.bar(covid_impactSouthAmerica, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs GINI Coefficient of Income in South America",
            color_continuous_scale='tealgrn'
            )
southamerica_exposure_income.show()

# #### Oceania

# In[37]:


#Oceania Data
covid_impactOceania = covid_impact2[(covid_impact2['Continent'] == 'Oceania')]
covid_impactOceania


# In[38]:


# Index of Exposure of Covid-19 infections and Index of Institutional Trust in Oceania: Bar Chart with Colored bar

oceania_exposure_trust = px.bar(covid_impactOceania, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Index of institutional trust'], color='Index of institutional trust',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index of Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs Index of Instituional Trust in Oceania",
            color_continuous_scale='sunset')
oceania_exposure_trust.show()


# Index Covid-19 Exposure in the Oceania Continent

oceania_exposure_income = px.bar(covid_impactOceania, x='Country name', y='Index of exposure to COVID-19  infections in other countries as of March 31',
             hover_data=['Gini coefficient of income'], color='Gini coefficient of income',
             labels={'Index of exposure to COVID-19  infections in other countries as of March 31':'Index Covid-19 Exposure'}, height=600, title= "Index Covid-19 Exposure vs GINI Coefficient of Income in Oceania",
            color_continuous_scale='tealgrn'
            )
oceania_exposure_income.show()

# In[39]:

#TOP 10 Countries: Covid-19 Deaths

data_10countries = covid_impact2.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries = data_10countries.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020')

fig_10countries = px.bar(datasort_10countries, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries.update_layout(title_text='Top10 Countries: Covid-19 Deaths')
fig_10countries.show()

#TOP 10 Countries of Europe: Covid-19 Deaths

data_10countries_europe = covid_impactEurope.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries_europe = data_10countries_europe.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020').head(10)

fig_10countries_europe = px.bar(datasort_10countries_europe, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries_europe.update_layout(title_text='Top10 Countries of Europe: Covid-19 Deaths')
fig_10countries_europe.show()



#TOP 10 Countries of Asia: Covid-19 Deaths

data_10countries_asia = covid_impactAsia.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries_asia = data_10countries_asia.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020').head(10)

fig_10countries_asia = px.bar(datasort_10countries_asia, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries_asia.update_layout(title_text='Top10 Countries of Asia: Covid-19 Deaths')
fig_10countries_asia.show()


#TOP 10 Countries of Africa: Covid-19 Deaths

data_10countries_africa = covid_impactAfrica.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries_africa = data_10countries_africa.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020').head(10)

fig_10countries_africa = px.bar(datasort_10countries_africa, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries_africa.update_layout(title_text='Top10 Countries of Africa: Covid-19 Deaths')
fig_10countries_africa.show()


#TOP 10 Countries of North America: Covid-19 Deaths

data_10countries_northamerica = covid_impactNorthAmerica.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries_northamerica = data_10countries_northamerica.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020').head(10)

fig_10countries_northamerica = px.bar(datasort_10countries_northamerica, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries_northamerica.update_layout(title_text='Top10 Countries of North America: Covid-19 Deaths')
fig_10countries_northamerica.show()



#TOP 10 Countries of South America: Covid-19 Deaths

data_10countries_southamerica = covid_impactSouthAmerica.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries_southamerica = data_10countries_southamerica.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020').head(10)

fig_10countries_southamerica = px.bar(datasort_10countries_southamerica, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries_southamerica.update_layout(title_text='Top10 Countries of South America: Covid-19 Deaths')
fig_10countries_southamerica.show()



#TOP 10 Countries of Oceania: Covid-19 Deaths

data_10countries_oceania = covid_impactOceania.loc[:,['Country name','COVID-19 deaths per 100,000 population in 2020','Continent']]

datasort_10countries_oceania = data_10countries_oceania.sort_values(ascending=False, by='COVID-19 deaths per 100,000 population in 2020').head(10)

fig_10countries_oceania = px.bar(datasort_10countries_oceania, 
                         x='Country name', 
                         y='COVID-19 deaths per 100,000 population in 2020',
                         color='Continent'
                         )
                         
fig_10countries_oceania.update_layout(title_text='Top10 Countries of Oceania: Covid-19 Deaths')
fig_10countries_oceania.show()





# In[41]:
continents_names = covid_impact2['Continent'].unique()
continents = [dict(label=continent ,value=continent_id) for continent, continent_id in zip(continents_names, covid_impact2.index)]
                                 
#fullnames = drivers['forename'] + str(" ") + drivers['surname']
#pilot_names = [dict(label=fullname, value=driver_id) for fullname, driver_id in zip(fullnames, drivers.index)]

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


#App itself

app = dash.Dash(__name__, 
             suppress_callback_exceptions=True,    
             external_stylesheets=[dbc.themes.DARKLY],)

server = app.server

app.title = "PROFILING COUNTRIES WITH RELATION TO COVID-19 DEATHS"

app.layout = html.Div([
    
    html.Div(children=[
    html.H1(children='PROFILING COUNTRIES WITH RELATION TO COVID-19 DEATHS'),

    ]),
    
    html.Div(dcc.Graph(id='bubble_age_deaths_graph',figure={bubble_age_deaths,'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }})),
    
    html.Div(dcc.Graph(id='pie_chart_femaleleaders',figure=pie_chart_female)),
    
    html.Div([
    
    html.Label(['Choose a Continent to analyse:'],style={'font-weight': 'bold'}),
    
    dcc.Dropdown(
    id='continent_drop1',
    options= continents,
    value=0,
    multi=False
    ),
    ]),
    
    html.Div(dcc.Graph(id='covid_deaths_graph',figure={})),

  
    html.Div([
    
    html.Label(['Choose a Continent to analyse:'],style={'font-weight': 'bold'}),
    
    dcc.Dropdown(
    id='continent_drop2',
    options= continents,
    value=0,
    multi=False
    ),
    ]),
    
    html.Div(dcc.Graph(id='exposure_trust_graph',figure={})),

    html.Div([
    
    html.Label(['Choose a Continent to analyse:'],style={'font-weight': 'bold'}),
    
    dcc.Dropdown(
    id='continent_drop3',
    options= continents,
    value=0,
    multi=False
    ),
    ]),
    
    html.Div(dcc.Graph(id='exposure_income_graph',figure={}))
    ])



@app.callback(
    Output('covid_deaths_graph', 'figure'),
   [Input(component_id='continent_drop1', component_property='value')]
)
def covid_death_continent(value):
    if value == 0:
        fig = fig_10countries_northamerica
        return fig
    elif value == 1:
        fig = fig_10countries_africa
        return fig
    elif value == 2:
        fig = fig_10countries_asia
        return fig
    elif value == 3:
        fig = fig_10countries_europe
        return fig
    elif value == 4:
        fig = fig_10countries_southamerica
        return fig
    else:
        fig = fig_10countries_oceania
        return fig

@app.callback(
    Output('exposure_trust_graph', 'figure'),
   [Input(component_id='continent_drop2', component_property='value')]
)
def exposure_trust_continent(value):
    if value == 0:
        fig = northamerica_exposure_trust
        return fig
    elif value == 1:
        fig = africa_exposure_trust
        return fig
    elif value == 2:
        fig = asia_exposure_trust
        return fig
    elif value == 3:
        fig = europe_exposure_trust
        return fig
    elif value == 4:
        fig = southamerica_exposure_trust
        return fig
    else:
        fig = oceania_exposure_trust
        return fig

@app.callback(
    Output('exposure_income_graph', 'figure'),
   [Input(component_id='continent_drop3', component_property='value')]
)
def exposure_income_continent(value):
    if value == 0:
        fig = northamerica_exposure_income
        return fig
    elif value == 1:
        fig = africa_exposure_income
        return fig
    elif value == 2:
        fig = asia_exposure_income
        return fig
    elif value == 3:
        fig = europe_exposure_income
        return fig
    elif value == 4:
        fig = southamerica_exposure_income
        return fig
    else:
        fig = oceania_exposure_income
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)

