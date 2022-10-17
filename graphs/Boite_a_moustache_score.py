# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:34:26 2022

@author: aeutarici
"""

import tkinter as tk
import pandas as pd
import plotly.graph_objects as go

#Importation du jeu de données et découpage en un plus petit jeu de données.
df = pd.read_csv("../datas/trainGraph.csv", sep=",")
df=df.drop_duplicates(subset=['iid'])
attrs_box=["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
df = df[attrs_box].rename(columns={'attr1_1':'Attirant', 'sinc1_1':'Sincere','intel1_1':'Intelligent','fun1_1': 'Fun','amb1_1':'Ambitieux', 'shar1_1':'Interets Communs'})

#Définition des couleurs de chaque boite a moustache
colors = ['rgba(240, 248, 255, 1 )', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

fig = go.Figure()
for xd, yd, cls in zip(df.columns, [df[i].to_list() for i in df.columns], colors):
        fig.add_trace(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker_size=2,
             #line_width=1
             )
        )

fig.update_layout(
    title='Importance de plusieurs critères lors du choix d\'un partenaire',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig.show()
