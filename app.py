import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split


def load_data():
    url = 'https://raw.githubusercontent.com/rfordatascience/' + \
    'tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv'
    df = pd.read_csv(url)
    df_interim = df.copy()
    df_interim = df_interim[['total_cup_points',
                                'species',
                                'country_of_origin',
                                'variety',
                                'aroma',
                                'aftertaste',
                                'acidity',
                                'body',
                                'balance',
                                'altitude_mean_meters',
                                "moisture"]]
    df_interim = df_interim.dropna()
    df_interim["species"] = pd.Categorical(df_interim["species"])
    df_interim["country_of_origin"] = pd.Categorical(df_interim["country_of_origin"])
    df_interim["variety"] = pd.Categorical(df_interim["variety"])
    df_interim["specialty"] = df_interim["total_cup_points"].apply(lambda x: "yes" if x > 82.43 else "no")
    df_interim["specialty"] = pd.Categorical(df_interim["specialty"])
    df_interim["altitude_mean_meters"] = df_interim["altitude_mean_meters"].apply(lambda x: 1300 if x > 10000 else x)
    df_interim = df_interim[df_interim["acidity"]!=0].copy()
    df = df_interim.copy()
    return df
    
df_ch = load_data()
st.write(df_ch.shape[0]) # 0 filas 1 columnas
st.title("Coffee Dashboard")
st.dataframe(df_ch)
# histograma de un valor
fig1 = px.histogram(df_ch,x="aroma")
st.plotly_chart(fig1)
fig2 = sns.pairplot(data=df_ch.drop(["species", "country_of_origin", "variety"], axis=1), hue="specialty")
st.pyplot(fig2)
df = df_interim.copy()
fig3 = df_ch.drop(["species", "country_of_origin", "variety"], axis=1).describe().T
df_train, df_test = train_test_split(df, test_size= 0.2, random_state= 2024, stratify=df['specialty'])

## x = st.slider("Select a value", min_value=-5, max_value=5, value=0)
## st.write(x,"Power of 3 is", x**2)
## y = st.slider("Select another value", min_value=-5, max_value=5, value=0)
## st.write(y,"Power o 3 is", y**3)

# Pintar EDA y hacer modelo que haga clasificación extraer la columna total_cup_points 

