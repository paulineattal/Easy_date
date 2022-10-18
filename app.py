from prepdatas import PrepDatas

import pandas as pd

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pickle
import base64
import io

from wordcloud import WordCloud

import gunicorn


app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title='EasyDate Michael Scott Team')
server = app.server 

df_graphes = pd.read_csv("./datas/trainGraph.csv", sep=",")
pds = PrepDatas(df_graphes)
pds.build_df_graphes()
df=pds.get_df()
df_men=pds.get_df_men()
df_women = pds.get_df_women()
df_boxplot = pds.get_df_boxplot()
df_word = pds.get_df_word()

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

wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(df_word)

plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.margins(0,0)
plt.savefig("./assets/word.png", bbox_inches = 'tight', pad_inches = 0)

loaded_model = pickle.load(open("model/model.pickle.dat", 'rb'))

#Menu
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
    html.Div([
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
    ], id="Accueil-tab"),

    #Page Statistique
    html.Div([
        html.H6("Graphique général pour l'exploration des données selon chaque variables", style=({"margin":"5px","text-align":"center"})),
        html.Div([
            dcc.Dropdown(id="xInput", options=[{"label":name,"value":name} for name in df.columns], value="age", style=({"width":"100%", "padding":"5px"})),
            dcc.Dropdown(id="yInput", options=[{"label":name,"value":name} for name in df.columns], value="income", style=({"width":"100%", "padding":"5px"})),
            dcc.Dropdown(id="colorInput", options=[{"label":name,"value":name} for name in df.columns], value="gender", style=({"width":"100%", "padding":"5px"}))
        ], id="DivFilter", style=({"display":"flex"})),
        dcc.Graph(id="GraphStat_1", style=({"width":"100%", "margin":"5px"})),
        html.H6("Représentation graphique des centres d'interêt par ages et revenus", style=({"margin":"5px","text-align":"center"})),
        html.Div([
            dcc.Graph(figure=px.sunburst(df_men, title="Hommes", path=['most_interest', 'goal_cat', 'age_cat'],
                    values='income', color='income', template="plotly_dark").update_layout(
                        margin=dict(l=5, r=5, t=50, b=5)
                    ), style=({"width":"50%", "margin":"5px"})),
            dcc.Graph(figure=px.sunburst(df_women, title="Femmes", path=['most_interest', 'goal_cat', 'age_cat'],
                    values='income', color='income', template="plotly_dark").update_layout(
                        margin=dict(l=5, r=5, t=50, b=5)
                    ), style=({"width":"50%", "margin":"5px"})),
        ], id="DivSunBurst", style=({"display":"flex"})),
        html.H6("Boxplot de l'importance accordée au différents critères demandés", style=({"margin":"5px","text-align":"center"})),
        dcc.Graph(figure=fig_plot, style=({"width":"100%", "margin":"5px"})),
        html.H6("Nuage de mot de l'importance accordées au différentes activités demandées", style=({"margin":"5px","text-align":"center"})),
        html.Img(src=r'assets/word.png', alt='image', style=({"display":"block","margin-left":"auto","margin-right":"auto","margin-top":"5px"})),
    ], id="Statistique-tab"),

    #Page Modélisation
    html.Div([
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
        html.Div([html.Img(src=r'assets/boosting_gradient.png', alt='image', height="400")],style={'marginBottom': 50, 'marginTop': 25,'text-align': 'center'})
    ], id="Modelisation-tab"),

    #Page prédiction
    html.Div([
        html.H5("Fichier de prédiction", style=({"margin-top":"5px"})),
        dcc.Upload(
            html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '100px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            id='uploadData',
        ),
        html.Div(id='outputPrediction'),
        html.Div([
            html.H5("Formulaire de prédiction"),
            dcc.Input(id="int_corr", type="number", placeholder="Score d'interet commun ('int_corr')", style=({"width":"72%", "margin":"5px"})),
            dcc.Input(id="age", type="number", placeholder="Age du participant ('age')", style=({"width":"35%", "margin":"5px"})),
            dcc.Input(id="age_o", type="number", placeholder="Age du partenaire ('age_o')", style=({"width":"35%", "margin":"5px"})),
            dcc.Input(id="attr1_1", type="number", placeholder="Note attribuée à l'attirance par le participant ('attr1_1')", style=({"width":"35%", "margin":"5px"})),
            dcc.Input(id="attr_o", type="number", placeholder="Note attribuée à l'attirance par le partenaire ('attr_o')", style=({"width":"35%", "margin":"5px"})),
            dcc.Input(id="fun1_1", type="number", placeholder="Note attribuée au fun par le participant ('fun1_1')", style=({"width":"35%", "margin":"5px"})),
            dcc.Input(id="fun_o", type="number", placeholder="Note attribuée au fun par le partenaire ('fun_o')", style=({"width":"35%", "margin":"5px"})),
            dcc.Input(id="income", type="number", placeholder="Revenu du participant ('income')", style=({"width":"72%", "margin":"5px"})),
            html.Div(html.Button('Valider', id='validFormPredict', style=({"border": "2px solid #fff", "width":"20%", "margin-top":"5px"})))
        ], id="formPrediction"),
        html.Div([
            html.H5("Prédiction", style=({"margin-top":"5px"})),
            html.P(id="predictionFromForm", style=({"margin-top":"5px"}))
        ], id="renderPrediction", style=({"display":"none"}))
    ], id="Prediction-tab")
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


@app.callback(Output('GraphStat_1', 'figure'),
    [Input('xInput', 'value'), Input('yInput', 'value'), Input('colorInput', 'value')])
def update_graph(xInput, yInput, colorInput):
    fig = px.scatter(df, x=xInput,
                     y=yInput,
                     color=colorInput, 
                     template="plotly_dark")
    return fig

@app.callback(Output('outputPrediction', 'children'),
              Input('uploadData', 'contents'))
def predFromFile(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        test = pd.read_csv(io.StringIO(decoded.decode('utf-8')))[['int_corr', 'age_o', 'age', 'attr_o', 'attr1_1', 'fun_o', 'fun1_1', 'income']]
        predictions = loaded_model.predict(test)
        children = [
            html.P(predictions)
        ]
        return children

@app.callback(
    [Output('predictionFromForm', 'children'), Output("renderPrediction", "style")],
    Input('validFormPredict', 'n_clicks'),
    [State('int_corr', 'value'), State('age_o', 'value'), State('age', 'value'), State('attr_o', 'value'), State('attr1_1', 'value'),
    State('fun_o', 'value'), State('fun1_1', 'value'), State('income', 'value')]
)
def predFromForm(n_clicks, int_corr, age_o, age, attr_o, attr1_1, fun_o, fun1_1, income):
    if n_clicks is None:
        raise PreventUpdate
    else:    
        test = pd.DataFrame({"int_corr":[int_corr],"age_o":[age_o],"age":[age],"attr_o":[attr_o],"attr1_1":[attr1_1],"fun_o":[fun_o],"fun1_1":[fun1_1],"income":[income]})
        predictions = loaded_model.predict(test)
        if predictions == [0]:
            predictions = "Pas match :("
        elif predictions == [1]:
            predictions = "Match :D !"
        children = [
            html.P(predictions, style=({"margin-top":"5px"}))
        ]
        return children, {"display":"block"}

if __name__ == '__main__':
    app.run_server(debug=True)
