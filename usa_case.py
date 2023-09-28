#!/usr/bin/env python
# coding: utf-8

# ## Case 2 house prices


# pip install geopy




# pip install pgeocode




import pandas as pd
# from geopy.geocoders import Nominatim
import streamlit as st
import pgeocode
import plotly.express as px
import numpy as np
# import time
import plotly.graph_objects as go
import statsmodels.api as sm


# df = pd.read_csv("usa_house.csv")
# df2 = pd.read_csv("kc_house.csv")
df = pd.read_csv(r"C:\Users\tjibb\Documents\school\jaar_3\minor\case_parijs\usa\usa_house.csv")
df2 = pd.read_csv(r"C:\Users\tjibb\Documents\school\jaar_3\minor\case_parijs\usa\kc_house.csv")

# In[148]:


df = df[df['price'] != 0]


# https://pypi.org/project/pgeocode/


df2.head()






# In[150]:


st.sidebar.markdown("## Inhoudsopgave")
st.sidebar.markdown("- [Sectie 1](#sectie-1)")
st.sidebar.markdown("- [Sectie 2](#sectie-2)")
st.sidebar.markdown("- [Sectie 2](#sectie-2)")


# https://docs.streamlit.io/library/api-reference/charts/st.map
# 
# https://plotly.com/python/bubble-maps/

# **te zien hoeveel en waar de huizen zijn**
# 
# * op de kaart zelf te zien 
# * ene kleur is voor de ene df en andere kleur voor de andere
# * het bouwjaar kan verschoven worden om alleen huizen in die bepaalde range te laten zien

# In[151]:


st.header("Sectie 1")

st.title("Verdeling van Huizen in Washington")

st.write(
    "In deze plot wordt de verdeling van de huizen weergegeven."
    " De lichtblauwe bolletjes vertegenwoordigen de data van de tweede dataframe,"
    " terwijl de donkerblauwe bolletjes de data van de eerste dataframe weergeven."
    " Op de kaart is te zien dat de huizen zich in de staat Washington bevinden."
    " Er zijn meerdere lichtblauwe bolletjes zichtbaar, omdat de data in deze dataframe gedetailleerder was."
    " Hierbij was de exacte locatie van de huizen bekend."
    " Voor de andere dataframe hebben we postcodes gebruikt om de locaties vast te leggen."
    " Hiervoor hebben we extra kolommen toegevoegd om de locatie te berekenen op basis van de postcodes."
)


# In[163]:


min_bouwjaar, max_bouwjaar = st.slider('Bouwjaren', 
                                      min_value=min(df['yr_built']), 
                                      max_value=max(df['yr_built']), 
                                      value=(min(df['yr_built']), max(df['yr_built'])))

# Filter de gegevens op basis van het bouwjaar
filtered_df = df[(df['yr_built'] >= min_bouwjaar) & (df['yr_built'] <= max_bouwjaar)]
filtered_df2 = df2[(df2['yr_built'] >= min_bouwjaar) & (df2['yr_built'] <= max_bouwjaar)]

fig = px.scatter_mapbox(filtered_df2, lat="lat", lon="long", 
                        color_discrete_sequence=["rgb(179,205,227)"])
fig.add_trace(px.scatter_mapbox(filtered_df, lat="latitude", lon="longitude", 
                                color_discrete_sequence=["rgb(55,126,184)"]).data[0])

# Opmaak van de kaart
fig.update_layout(
    mapbox_style="carto-positron",  
    autosize=True
)

# Toon de kaart
# fig.show()

st.plotly_chart(fig)


# **Gemiddelde huisprijzen per stad per jaar**
# 
# * drop down optie met stad
# * x-as is het jaar
# * y-as is de gemiddelde prijs

# In[153]:


st.header('Sectie 2')

st.title("Gemiddelde Huisprijzen per Jaar")

st.write(
    "In de volgende plot zijn de gemiddelde huisprijzen per jaar te zien."
    " Hierbij zijn beide dataframes in één plot samengevoegd."
    " Via de dropdown-box kan er een stad worden geselecteerd om naar te kijken."
    " Er is alleen gefilterd op steden die in beide dataframes voorkomen."
    " Voor de tweede dataframe is er een kolom toegevoegd om de locatie om te zetten naar de desbetreffende stad."
)


# In[164]:


# Titel en beschrijving


# Dropdown-menu voor steden
steden_df1 = set(df['city'])
steden_df2 = set(df2['city'])
gemeenschappelijke_steden = steden_df1.intersection(steden_df2)
selected_city = st.selectbox('Selecteer een stad', gemeenschappelijke_steden)

# Filter de dataset op de geselecteerde stad
filtered_data = df[df['city'] == selected_city]
filtered_data2 = df2[df2['city'] == selected_city]


# Groepeer de data per jaar en stad en bereken het gemiddelde per jaar
avg_price_per_year = filtered_data.groupby(['yr_built', 'city'])['price'].mean().reset_index()
avg_price_per_year2 = filtered_data2.groupby(['yr_built', 'city'])['price'].mean().reset_index()


# Maak de Plotly-lijngrafiek
fig = go.Figure()

fig.add_trace(go.Scatter(x=avg_price_per_year["yr_built"], y=avg_price_per_year["price"], mode='lines', name='Dataframe 1 USA', line=dict(color='salmon')))
fig.add_trace(go.Scatter(x=avg_price_per_year2["yr_built"], y=avg_price_per_year2["price"], mode='lines', name='Dataframe 2 King County', line=dict(color='steelblue')))

fig.update_layout(
    title=f"Prijsontwikkeling in {selected_city}",
    xaxis_title="Jaar",
    yaxis_title="Prijs",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)


st.plotly_chart(fig)


# In[160]:




st.title("Prijs versus Bouwjaar van Huizen")

st.write(
    "In de volgende plot is de prijs tegenover het bouwjaar van huizen geplaatst."
    " Er is onderscheid gemaakt tussen gerenoveerde en niet-gerenoveerde huizen."
    " Een checkbox is toegevoegd waarmee je de plot met of zonder uitschieters (uitbijters) kunt bekijken."
    " In de legenda wordt aangegeven welke kleur wordt gebruikt voor gerenoveerde en niet-gerenoveerde huizen."
    " Met de dropdown-box kun je de gewenste dataset kiezen."
)


# In[161]:


#verwijderen van uitbijters
def remove_outliers(df, threshold=2):
    z_scores = np.abs((df - df.mean()) / df.std())
    return df[(z_scores < threshold).all(axis=1)]

# Streamlit-applicatie
st.title('Scatter Plot van Prijs vs. Bouwjaar')

dataset_keuze = st.selectbox('Selecteer een dataset', ['Dataset 1 USA', 'Dataset 2 King County'])

st.markdown('In de volgende plot is de prijs tegenover het bouwjaar gezet, hierbij is er een verschil gemaakt in wel of niet gerenoveerde huizen')

# Voeg een dropdown-menu toe voor het filteren van uitbijters
show_outliers = st.checkbox('Toon uitbijters', key='show_outliers', value=False)

# Filter de data op basis van de geselecteerde optie
if show_outliers:
    filtered_data = df[['yr_built', 'price', 'yr_renovated']]
    filtered_data2 = df2[['yr_built', 'price', 'yr_renovated']]
else:
    filtered_data = remove_outliers(df[['yr_built', 'price', 'yr_renovated']])
    filtered_data2 = remove_outliers(df2[['yr_built', 'price', 'yr_renovated']], 5)
    
# Maak een nieuwe kolom om het renovatiestatus weer te geven
filtered_data['Renovatie_Status'] = filtered_data['yr_renovated'].apply(lambda x: 'Gerennoveerd' if x != 0 else 'Niet Gerennoveerd')
filtered_data2['Renovatie_Status'] = filtered_data2['yr_renovated'].apply(lambda x: 'Gerennoveerd' if x != 0 else 'Niet Gerennoveerd')

#colorpaltete
color_palet = {'Gerennoveerd':'salmon', 'Niet Gerennoveerd':'steelblue'}

# Scatter plot met Plotly Express
if dataset_keuze == 'Dataset 1 USA':
    fig = px.scatter(filtered_data, x='yr_built', y='price', color='Renovatie_Status', color_discrete_map=color_palet, title='Scatter Plot van Prijs vs. Bouwjaar')
else:
    fig = px.scatter(filtered_data2, x='yr_built', y='price', color='Renovatie_Status', color_discrete_map=color_palet, title='Scatter Plot van Prijs vs. Bouwjaar')
    fig.update_layout(legend=dict(traceorder = 'reversed'))
    fig.update_xaxes(title_text='Bouwjaar')
fig.update_yaxes(title_text='Prijs')

# Toon de plot in Streamlit
st.plotly_chart(fig)


# In[157]:


df['numberofrooms'] = df['bedrooms'] + df['bathrooms'] + 1
df2['numberofrooms'] = df2['bedrooms'] + df2['bathrooms'] + 1
df['price p area'] = df['price'] / df['sqft_lot']
df = df[df['price'] != 0]
df2 = df2[df2['price'] != 0]


# In[158]:


# def verwijder_uitbijters(df, var):
#     Q3 = df[var].quantile(0.75)
#     Q1 = df[var].quantile(0.25)
#     IQR = Q3-Q1
#     upper = Q3 + (1.5 * IQR)
#     lower = Q1 - (1.5 * IQR)

#     df = df[(df[var] > lower) & (df[var] < upper)]

#     return df
    
# # Lijst van kolommen om uitbijters te verwijderen
# columns_to_process = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#         'condition', 'sqft_above', 'sqft_basement']

# # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df'
# for column in columns_to_process:
#     df_cor = verwijder_uitbijters(df, column)

# # Lijst van kolommen om uitbijters te verwijderen in df2
# columns_to_process_df2 = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
#        'sqft_lot', 'floors', 'condition', 'grade',
#        'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

# # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df2'
# for column in columns_to_process_df2:
#     df_cor2 = verwijder_uitbijters(df2, column)

# Nu zijn 'df' en 'df2' bijgewerkt met de verwijderde uitbijters



# In[159]:


st.header("Sectie 3")

# Titel
st.title("Correlatietabel voor 'Price' vs Andere Variabelen")

# Inleidingstekst
st.write(
    "In de onderstaande tabel is de correlatie weergeven van price met de andere variabelen uit de dataset. "
    "Voordat de correlatie wordt berekend, zijn de uitbijters verwijderd door een ondergrens en bovengrens te bepalen "
    "met behulp van de kwartielafstand. Door middel van de knopen kan er gewisseld worden van dataset."
)

# Alinea over correlatie met sqft_living
st.write(
    "In beide datasets is te zien dat 'sqft_living' de grootste correlatie heeft in vergelijking met de andere variabelen. "
    "Daarom wordt er met deze variabele een voorspelling gemaakt van de price aan de hand van lineaire regressie."
)


# In[162]:




def verwijder_uitbijters(df, var):
    Q3 = df[var].quantile(0.75)
    Q1 = df[var].quantile(0.25)
    IQR = Q3-Q1
    upper = Q3 + (1.5 * IQR)
    lower = Q1 - (1.5 * IQR)

    df = df[(df[var] > lower) & (df[var] < upper)]

    return df



# Dropdown-menu voor weergave met of zonder uitbijters
show_outliers33 = st.checkbox('Toon uitbijters', key='show_outliers33', value=False)

# Verwijder uitbijters op basis van de geselecteerde kolom en weergaveoptie
if show_outliers33:
    df_cor = df
    df_cor2 = df2
else:
    # Lijst van kolommen om uitbijters te verwijderen
    columns_to_process = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'condition', 'sqft_above', 'sqft_basement']

    #Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df'
    for column in columns_to_process:
        df_cor = verwijder_uitbijters(df, column)

    # Lijst van kolommen om uitbijters te verwijderen in df2
    columns_to_process_df2 = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

    # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df2'
    for column in columns_to_process_df2:
        df_cor2 = verwijder_uitbijters(df2, column)
        

# Kies de kolom waarvoor je de correlatie wilt berekenen
column_to_correlate = 'price'

# Bereken de correlatiecoëfficiënten tussen de geselecteerde kolom en andere numerieke kolommen
correlation_matrix = df_cor.corr()
correlation_matrix2 = df_cor2.corr()

# Kies alleen de correlatie van de geselecteerde kolom met andere kolommen
correlation_with_target = correlation_matrix[column_to_correlate]
correlation_with_target2 = correlation_matrix2[column_to_correlate]
# Maak een bar plot om de correlatie te visualiseren
fig1 = px.bar(
    x=correlation_with_target.index,
    y=correlation_with_target.values,
    labels={'x': 'Kolom', 'y': f'Correlatie met {column_to_correlate}'},
    title=f'Correlatie met {column_to_correlate}',
    color=correlation_with_target.values
)
fig1.update_xaxes(tickangle=90)

# Maak een bar plot om de correlatie te visualiseren
fig2 = px.bar(
    x=correlation_with_target2.index,
    y=correlation_with_target2.values,
    labels={'x': 'Kolom', 'y': f'Correlatie met {column_to_correlate}'},
    title=f'Correlatie met {column_to_correlate}',
    color=correlation_with_target2.values
)
fig2.update_xaxes(tickangle=90)

fig1_button = {'method': 'update', 'label': 'Figure 1', 'args': [{'visible': [True, False]}, {'title': 'Figure 1'}]}
fig2_button = {'method': 'update', 'label': 'Figure 2', 'args': [{'visible': [False, True]}, {'title': 'Figure 2'}]}



# Voeg knoppen toe voor de figuren
button_1 = st.button('Data 1')
button_2 = st.button('Data 2')


if button_1:
    st.plotly_chart(fig1)
elif button_2:
    st.plotly_chart(fig2)




# Titel
st.title("Regressie")

# Inleidingstekst
st.write(
    "In de onderstaande grafiek is een scatterplot weergegeven met een regressielijn. "
    "Door middel van het dropdownmenu kan er een variabele geselecteerd worden om te vergelijken met de variabele 'price'. "
    "Door gebruik te maken van de knoppen kan er gewisseld worden tussen datasets en kun je de regressielijnen vergelijken tussen de verschillende datasets. "
    "Tenslotte kun je ook vergelijken of de uitbijters van invloed zijn op de regressie door uitbijters aan of uit te zetten."
)

# Alinea over R2-score en sqft_living
st.write(
    "In beide datasets is te zien dat de variabele 'sqft_living' de beste R2-score geeft. "
    "Dit was te verwachten aangezien deze variabele ook de grootste correlatie had met 'price'. "
    "Echter valt wel op dat als je uitbijters weghaalt bij dataset 1, de R2-score verbetert. "
    "Maar dit is niet het geval bij dataset 2; als je hier de uitbijters weghaalt, wordt de R2-score slechter."
)

# Voeg hier je dropdownmenu, knoppen en andere interactieve elementen toe




def verwijder_uitbijters(df, var):
    Q3 = df[var].quantile(0.75)
    Q1 = df[var].quantile(0.25)
    IQR = Q3-Q1
    upper = Q3 + (1.5 * IQR)
    lower = Q1 - (1.5 * IQR)

    df = df[(df[var] > lower) & (df[var] < upper)]

    return df

df_sqft = df[['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']]
df_sqft2 = df2[['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']]

# Dropdown-menu voor kenmerken op de x-as
selected_feature = st.selectbox('Selecteer een kenmerk voor de x-as', df_sqft.columns)

# Dropdown-menu voor weergave met of zonder uitbijters
show_outliers666 = st.checkbox('Toon uitbijters', key='show_outliers666', value=False)

# Verwijder uitbijters op basis van de geselecteerde kolom en weergaveoptie
if show_outliers666:
    df = df
    df2 = df2
else:
    # Lijst van kolommen om uitbijters te verwijderen
    columns_to_process = ['price', selected_feature]

    # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df'
    for column in columns_to_process:
        df = verwijder_uitbijters(df, column)

    # Lijst van kolommen om uitbijters te verwijderen in df2
    columns_to_process_df2 = ['price', selected_feature]

    # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df2'
    for column in columns_to_process_df2:
        df2 = verwijder_uitbijters(df2, column)

    
# Maak een scatterplot met OverallQual op de x-as en SalePrice op de y-as
fig1 = px.scatter(df, x=selected_feature, y='price')
fig2 = px.scatter(df2, x=selected_feature, y='price')

fig1.update_traces(marker=dict(color='green'))
fig2.update_traces(marker=dict(color='blue'))

# # Stel de kleur van de trendlijn in op rood (R)
# fig1.update_traces(line=dict(color='red'))
# fig2.update_traces(line=dict(color='red'))

fig1_button = {'method': 'update', 'label': 'Figure 1', 'args': [{'visible': [True, False]}, {'title': 'Figure 1'}]}
fig2_button = {'method': 'update', 'label': 'Figure 2', 'args': [{'visible': [False, True]}, {'title': 'Figure 2'}]}



# Voeg knoppen toe voor de figuren
button_3 = st.button('Dataset 1')
button_4 = st.button('Dataset 2')


if button_3:
    st.plotly_chart(fig1)
elif button_4:
    st.plotly_chart(fig2)




