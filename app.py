from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import gunicorn 
from prepdata import PrepData

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='EasyDate Michael Scott Team')
server = app.server 

#get all df from PrepData class for graphs
df_graphes = pd.read_csv("./datas/trainGraph.csv", sep=",")
pd = PrepData(df_graphes)
pd.build_df_graphes()
df=pd.get_df()
df_men=pd.get_df_men()
df_women = pd.get_df_women()


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
        dcc.Graph(id="GraphStat_3")
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


@app.callback(Output('GraphStat_1','figure'), Output('GraphStat_2','figure'), Output("GraphStat_3", "figure"),
    [Input('xInput', 'value'), Input('yInput', 'value'), Input('colorInput', 'value')])
def update_graph(xInput, yInput, colorInput):
    
    dfg = df.groupby(by=[xInput,colorInput])[yInput].mean().reset_index()
    dfg[colorInput] = dfg[colorInput].astype(str)
    fig = px.bar(dfg, x=xInput,
                      y=yInput,
                      color=colorInput,
                      barmode="group")
    fig_men = px.sunburst(df_men, path=['most_interest', 'goal_cat', 'age_cat'],
                  values='income', color='income'
                 )
    fig_women = px.sunburst(df_women, path=['most_interest', 'goal_cat', 'age_cat'],
                  values='income', color='income'
                 )

    return fig, fig_men, fig_women


if __name__ == '__main__':
    app.run_server(debug=True)
