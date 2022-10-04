# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



# Import librairies
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


#Instanciation de l'odjet dash
app = dash.Dash(__name__)
server = app.server


#Import dataset
df = pd.read_csv("./clean.csv", sep=",")

#compter les match par vagues
match_wave = df[["wave", "match"]].groupby("wave").sum("match").reset_index()


#Présentation du nombre de visite total de chaque monument regroupé par sites
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

fig = px.bar(match_wave, x="wave", y="match", color="wave", barmode="group") #typage du graphe

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Nombre de matchs de chaque vague',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),

    html.Div(children='Présentation du nombre de visite total de chaque monument regroupé par sites', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='Nb_vistes_par_monument-sites',
        figure=fig
    )
])


#Sortie
app.run_server(debug=True, port=8051)

