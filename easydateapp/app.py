from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='EasyDate Michael Scott Team')

df = pd.read_csv("trainClean.csv")

#CSS placement
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

fig = px.scatter(df, x="age", y="income", color="gender")
#SideBar
SideBar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Accueil", href="/", active="exact"),
                dbc.NavLink("Statistique", href="/Statistique", active="exact"),
                dbc.NavLink("Modélisation", href="/Modelisation", active="exact"),
                dbc.NavLink("Prédiction", href="/Prediction", active="exact"),                
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE
)
#Contenu
PageContent = html.Div(id="page-content", style=CONTENT_STYLE)

#Apparence
app.layout = html.Div([dcc.Location(id="url"), SideBar, PageContent])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("Accueil de Légende")
    elif pathname == "/Statistique":
        return dcc.Graph(id='example-graph-2', figure=fig)
    elif pathname == "/Modelisation":
        return html.P("Ici on modélise")
    elif pathname == "/Prediction":
        return html.P("Vla la prediction en mode API rest")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == '__main__':
    app.run_server(debug=True)
