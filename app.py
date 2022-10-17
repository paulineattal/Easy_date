from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import gunicorn 
from prepdatas import PrepDatas
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='EasyDate Michael Scott Team')
server = app.server 

#get all df from PrepData class for graphs
df_graphes = pd.read_csv("./datas/trainGraph.csv", sep=",")
pds = PrepDatas(df_graphes)
pds.build_df_graphes()
df=pds.get_df()
df_men=pds.get_df_men()
df_women = pds.get_df_women()
df_boxplot = pds.get_df_boxplot()


#SideBar
TABPANEL = dbc.Container([
    html.H1("Menu"),
    html.Hr(),
    dbc.Tabs(
        [
            dbc.Tab(label="Accueil", tab_id="Accueil"),
            dbc.Tab(label="Statistique", tab_id="Statistique"),
            dbc.Tab(label="Modélisation", tab_id="Modelisation"),
            dbc.Tab(label="Prédiction", tab_id="Prediction")             
        ],
        id="tabPanel",
        active_tab="Accueil",
    )
])

#Contenu
PageContent = dbc.Container([

    #Accueil
    html.Div(id="Accueil-tab", children=
        html.Div(
            html.P("Accueil")
        )
    ),

    #Page Statistique
    html.Div(id="Statistique-tab", children=[
        html.Div([
            dcc.Dropdown(id="xInput", options=[{"label":name,"value":name} for name in df.columns], value="age", className="app-DivFilter--Select_value"),
            dcc.Dropdown(id="yInput", options=[{"label":name,"value":name} for name in df.columns], value="income", className="app-DivFilter--Select_value"),
            dcc.Dropdown(id="colorInput", options=[{"label":name,"value":name} for name in df.columns], value="gender", className="app-DivFilter--Select_value")], 
            className="DivFilter"
        ),
        dcc.Graph(id="GraphStat_1"),
        dcc.Graph(id="GraphStat_2"),
        dcc.Graph(id="GraphStat_3"),
        dcc.Graph(id="GraphStat_4")
    ], className="DivTab"),

    #Page Modélisation
    html.Div(id="Modelisation-tab", children=
        html.Div(
            html.P("Modelisation")
        )
    ),

    #Page prédiction
    html.Div(id="Prediction-tab", children=
        html.Div(
            html.P("Prediction")
        )
    )
])


#Apparence
app.layout = html.Div([TABPANEL, PageContent])

@app.callback([Output("Accueil-tab", "style"), Output("Statistique-tab", "style"), Output("Modelisation-tab", "style"), Output("Prediction-tab", "style")],
              [Input("tabPanel","active_tab")])
def render_tab_content(active_tab):
    on = {"display":"block"}
    off = {"display":"none"}
    if active_tab is not None:
        if active_tab == "Accueil":
            return on, off, off, off
        elif active_tab == "Statistique":
            return off, on, off, off
        elif active_tab == "Modelisation":
            return off, off, on, off
        elif active_tab == "Prediction":
            return off, off, off, on
    return "No tab selected"


@app.callback(Output('GraphStat_1','figure'), Output('GraphStat_2','figure'), Output("GraphStat_3", "figure"),  Output("GraphStat_4", "figure"),
    [Input('xInput', 'value'), Input('yInput', 'value'), Input('colorInput', 'value')])
def update_graph(xInput, yInput, colorInput):
    
    ###graphbar###
    dfg = df.groupby(by=[xInput,colorInput])[yInput].mean().reset_index()
    dfg[colorInput] = dfg[colorInput].astype(str)
    fig = px.bar(dfg, x=xInput,
                      y=yInput,
                      color=colorInput,
                      barmode="group")
    
    ###sunburst###
    fig_men = px.sunburst(df_men, path=['most_interest', 'goal_cat', 'age_cat'],
                  values='income', color='income'
                 )
    fig_women = px.sunburst(df_women, path=['most_interest', 'goal_cat', 'age_cat'],
                  values='income', color='income'
                 )
    
    ###boxplot###
    fig_plot = go.Figure()
    for xd, yd, cls in zip(df_boxplot.columns, [df_boxplot[i].to_list() for i in df_boxplot.columns], ['rgba(240, 248, 255, 1 )', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)','rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']):
            fig_plot.add_trace(go.Box(
                y=yd,
                name=xd,
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                fillcolor=cls,
                marker_size=2
                 )
            )
    fig_plot.update_layout(
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
    ###graph words###
    

    return fig, fig_men, fig_women, fig_plot


if __name__ == '__main__':
    app.run_server(debug=True)
