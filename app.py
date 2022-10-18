from prepdatas import PrepDatas
from fig import Fig

import pandas as pd

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import plotly.express as px

import pickle
import base64
import io

import gunicorn


app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], title='EasyDate Michael Scott Team')
server = app.server


df_graphes = pd.read_csv("./datas/trainGraph.csv", sep=",")
pds = PrepDatas(df_graphes)
pds.build_df_graphes()
df=pds.get_df()
df_men=pds.get_df_men()
df_women = pds.get_df_women()
df_boxplot = pds.get_df_boxplot()
df_word = pds.get_df_word()

df_dropdown_x_y = df[['int_corr', 'age_o','attr_o', 'sinc_o','intel_o', 'fun_o', 'amb_o', 'shar_o', 
                      'age', 'imprace', 'imprelig', 'income', 'sports', 'tvsports', 'exercise', 
                      'dining', 'museums','art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 
                      'theater','movies', 'concerts', 'music', 'shopping', 'yoga', 'exphappy', 
                      'expnum','attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']].rename(columns={"int_corr":"score_interet_commun", "age_o":"age_p1","attr_o":"score_attirance_p1", 
                 "sinc_o":"score_sincerite_p1","intel_o":"score_intelligence_p1", "fun_o":"score_fun_p1","amb_o":"score_ambition_p1",
                 "shar_o":"score_interet_commun_p1", "imprace":"importance_origine_commune","imprelig":"importance_religion_commune",
                 "income":"salaire","exphappy":"retour_experience", "expnum":"confiance_experience",
                 'attr1_1':"score_attirance_p2", 'sinc1_1':"score_sincerite_p2", 'intel1_1':"score_intelligence_p2", 'fun1_1':"score_fun_p2", 'amb1_1':"score_ambition_p2", 'shar1_1':"score_interet_commun_p2"})


df_dropdown_mod = df[['gender', 'wave', 'round','match', 'race_o', 'field_cd', 
                      'race', 'goal', 'date', 'go_out']].rename(columns={"gender":"sexe","wave":"vague",
                                                                                "race_o":"origine_p1","goal":"objectif",
                                                                                "field_cd":"code_carriere","race":"origine_p2",
                                                                                "date":"freq_rdv_amoureux","go_out":"freq_sortie"})

fig = Fig(df,df_men,df_women,df_boxplot,df_word)
fig_sunburst_men = fig.get_fig_sunburst_men()
fig_sunburst_women = fig.get_fig_sunburst_women()
fig_boxplot = fig.get_fig_boxplot()
fig.fig_words()


loaded_model = pickle.load(open("model/model.pickle.dat", 'rb'))

#Menu
TABPANEL = dbc.Container([
    html.H1("AI Match X Easy Date", style=({"margin":"5px","text-align":"center"})),
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
            html.P("Bienvenue sur l'application AI Match X Easy Date",style={'margin':'20px','textAlign': 'center','color': 'pink','fontSize': 30})), 
        html.Div([
            html.P("Easy Date",style={'textAlign': 'left','color': 'pink','fontSize': 20})]),
        dbc.Row([
            dbc.Col(html.Img(src=r'assets/logo.png', alt='image', width="200"), width=2),
            dbc.Col(html.P(["Easy Date est une société d'événementiel qui organise des speed dating. Lors des 17 vagues de speed dating, 452 célibataires ont tenté de trouver l'amour!", html.Br(),
                    "De nombreuses données ont été récoltés pendant ces speed dating et Easy Date voudrait un modèle prédictif pour savoir si deux personnes sont compatibles prior à leur rencontre."], style={'textAlign': 'left'}), width=10),
                ]),
        html.Div([
            html.P("AI Match",style={'textAlign': 'left','color': 'pink','fontSize': 20})]),
        dbc.Row([
            dbc.Col(html.Img(src=r'assets/logo2.png', alt='image', width="180"), width=2),
            dbc.Col( html.P([html.Br(),"AI Match est une équipe formée de 4 data scientists spécialisés dans les modèles prédictifs.", html.Br(),
                     " Grâce aux données nos data scientists ont réussi à répondre à la demande du client. Nous allons maintenant pouvoir simplement entrer les variables d'interet dans un formulaire et découvrir si le grand amour est au rendez vous "], style={'textAlign': 'left'}),width=10)])
                ]),        

    #Page Statistique
    html.Div([
        html.H4("Graphique général pour l'exploration des données selon chaque variables", style=({"margin":"15px","text-align":"center"})),
        html.Div([
            dcc.Dropdown(id="xInput", options=[{"label":name,"value":name} for name in df_dropdown_x_y.columns], value="age", style=({"width":"100%", "padding":"5px"})),
            dcc.Dropdown(id="yInput", options=[{"label":name,"value":name} for name in df_dropdown_x_y.columns], value="income", style=({"width":"100%", "padding":"5px"})),
            dcc.Dropdown(id="colorInput", options=[{"label":name,"value":name} for name in df_dropdown_mod.columns], value="gender", style=({"width":"100%", "padding":"5px"}))
        ], id="DivFilter", style=({"display":"flex"})),
        dcc.Graph(id="GraphStat_1", style=({"width":"100%", "margin":"5px"})),
        html.H4("Représentation graphique des centres d'interêt par ages et revenus", style=({"margin":"15px","text-align":"center"})),
        html.Div([
            dcc.Graph(figure=fig_sunburst_men, style=({"width":"50%", "margin":"5px"})),
            dcc.Graph(figure=fig_sunburst_women, style=({"width":"50%", "margin":"5px"})),
        ], id="DivSunBurst", style=({"display":"flex"})),
        html.H4("Boxplot de l'importance accordée au différents critères demandés", style=({"margin":"15px","text-align":"center"})),
        dcc.Graph(figure=fig_boxplot, style=({"width":"100%", "margin":"5px"})),
        html.H4("Nuage de mot de l'importance accordées au différentes activités demandées", style=({"margin":"15px","text-align":"center"})),
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
        dcc.Download(id="download-pred"),
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
                     color=colorInput)
    fig.update_layout({'plot_bgcolor':'rgb(39, 43, 48)', 'paper_bgcolor':'rgb(39, 43, 48)'})
    return fig

@app.callback(Output("download-pred", 'data'),
              Input('uploadData', 'contents'))
def predFromFile(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        test = pd.read_csv(io.StringIO(decoded.decode('utf-8')))[['int_corr', 'age_o', 'age', 'attr_o', 'attr1_1', 'fun_o', 'fun1_1', 'income']]
        predictions = pd.Series(loaded_model.predict(test))
        return dcc.send_data_frame(predictions.to_csv, "predictions.csv")

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
            predictions = "💔"
        elif predictions == [1]:
            predictions = "❤️"
        children = [
            html.H1(predictions, style=({"margin-top":"5px"}))
        ]
        return children, {"display":"block"}

if __name__ == '__main__':
    app.run_server(debug=True)
