# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:34:26 2022

@author: aeutarici
"""

import tkinter as tk
import pandas as pd
import plotly.graph_objects as go
import numpy as np

df = pd.read_csv("../datas/trainGraph.csv", sep=",")
df1=df.drop_duplicates(subset=['iid'])
X=df1[["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]]



x_data = ['Attirant', 'Sincere',
          'Intelligent', 'Fun',
          'Ambitieux', 'Interets Communs',]

y0 = df1['attr1_1'].astype(np.int)
y1 = df1['sinc1_1'].astype(np.int)
y2 = df1['intel1_1'].astype(np.int)
y3 = df1['fun1_1'].astype(np.int)
y4 = df1['amb1_1'].astype(np.int)
y5 = df1['shar1_1'].astype(np.int)
y_data = [y0, y1, y2, y3, y4, y5]

colors = ['rgba(240, 248, 255, 1 )', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

fig = go.Figure()

for xd, yd, cls in zip(x_data, y_data, colors):
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