from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
#import gunicorn 

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], title='EasyDate Michael Scott Team')
server = app.server 

df = pd.read_csv("./datas/trainClean.csv")
df_2 = pd.read_csv("./datas/trainGraph.csv", sep=",")

conditions = [
    (df_2['goal'] == 1) | (df_2['goal'] == 2),
    (df_2['goal'] == 4) | (df_2['goal'] == 3),
    (df_2['goal'] == 5) | (df_2['goal'] == 6)
    ]
values = ['fun', 'serious', 'pass time']
df_2['goal_cat'] = np.select(conditions, values)
conditions = [
    (df_2['age'] > min(df_2["age"])) & (df_2['age'] <= 22),
    (df_2['age'] > 22) & (df_2['age'] <= 30),
    (df_2['age'] > 30)
    ]
values = ['young', 'adult', 'old']

df_2['age_cat'] = np.select(conditions, values)
df_2["culture_interest"]=np.where(df_2['culture']>=6, 'yes_culture', 'no_culture')
df_2["indoors_interest"]=np.where(df_2['indoors']>=6, 'yes_indoors', 'no_indoors')
df_2["sport_interest"]=np.where(df_2['sport']>=6, 'yes_sport', 'no_sport')
df_2["social_interest"]=np.where(df_2['social']>=6, 'yes_social', 'no_social')
df_2["most_interest"]=df_2[["culture","indoors","sport","social"]].idxmax(axis=1)
df_2["gender"]=df_2["gender"].replace(0, "women")
df_2["gender"]=df_2["gender"].replace(1, "men")
df_men = df_2[df_2["gender"]=="men"]
df_women = df_2[df_2["gender"]=="women"]

#SideBar
TABPANEL = dbc.Container([
    html.H1("AI Match X Easy Date"),
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
    html.Div(id="Accueil-tab", children=[
        html.Div(
            html.P("Bienvenue sur l'application AI Match X Easy Date",style={'textAlign': 'center','color': 'pink','fontSize': 30})),
        html.Div([
            html.P("Easy Date",style={'textAlign': 'left','color': 'pink','fontSize': 20}),
            html.P(["Easy Date est une société d'événementiel qui organise des speed dating. Lors des 17 vagues de speed dating, 452 célibataires ont tenté de trouver l'amour!", html.Br(),
                    "De nombreuses données ont été récoltés pendant ces speed dating et Easy Date voudrait un modèle prédictif pour savoir si deux personnes sont compatibles prior à leur rencontre "], style={'textAlign': 'left'}),
            html.Img(src=r'assets/logo.png', alt='image', width="200"),
        ],style={'marginBottom': 50, 'marginTop': 25,'text-align': 'right'}),
        html.Div([
            html.P("AI Match",style={'textAlign': 'left','color': 'pink','fontSize': 20}),
            html.P(["AI Match est une équipe formée de 4 data scientists spécialisés dans les modèles prédictifs.", html.Br(),
                    " Grâce aux données nos data scientists ont réussi à répondre à la demande du client."], style={'textAlign': 'left'}),
            html.Img(src=r'assets/logo2.png', alt='image', width="200"),
        ],style={'marginBottom': 50, 'marginTop': 25,'text-align': 'right'})

        ]),


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
    html.Div(id="Modelisation-tab", children=[
        html.Div(
            html.P("Explication du modèle prédictif",style={'textAlign': 'center','color': 'pink','fontSize': 30})),
        html.Div([
            html.P("Boosting",style={'color': 'pink','fontSize': 20}),
            html.P(["Le boosting est une méthode qui permet de transformer les apprenants faibles en apprenants forts. La procédure commence par former des arbres de décision. Chaque observation  se voit attribuer un poids égal", html.Br(),                
                    "Après avoir analysé le premier arbre, on augmente le poids de chaque observation difficle à classer et on diminue le poids des observations qui n'ont pas posé de problème.",html.Br(),
                    "Le prochain arbre est donc construit sur les données pondérées ce qui améliore les prévisions du premier arbre."]),
                ]),
        html.Div([html.Img(src=r'assets/boosting.png', alt='image', height="400")],style={'marginBottom': 50, 'marginTop': 25,'text-align': 'center'}),
        html.Div([
            html.P("Boosting de gradient",style={'color': 'pink','fontSize': 20}),
            html.P(["Le boosting de gradient est une catégorie de boosting.", html.Br(),                
                    "Il repose fortement sur la prédiction que le prochain modèle réduira les erreurs de prédiction lorsqu’il sera mélangé avec les précédents. L’idée principale est d’établir des résultats cibles pour le prochain modèle afin de minimiser les erreurs.",html.Br(),
                    ""])]),
        html.Div([html.Img(src=r'assets/boosting_gradient.png', alt='image', height="400")],style={'marginBottom': 50, 'marginTop': 25,'text-align': 'center'}),
        ]
           
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
