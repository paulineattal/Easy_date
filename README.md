# Easy_date
Repos du projet *Easy Date* du cours de python M2. Application déployé en Python avec l'utilisation de la librairie Dash. Cette application présente des graphiques interractifs Plotly. 
Cette application est hebergé sur le serveur Heroku et peut etre visualisé via le lien https://easy-date-micheal-scott-team.herokuapp.com/

## Aperçu de ce que fait chaque fichier
Plusieurs fichiers sont spécifiques et necessaires pour le fonctionnement de l'application sur le serveur Heroku. 

`app.py` où l'application dash est depployé, et où tous les graphiques plotly sont créés <br>
`prepdata.py` contient une classe et des fonctions de preparation du Dataframe pour les graphiques <br>
`requirements.txt` contenannt les modules python qui seront installés lors de la construction de l'application <br>
`runtime.txt` indique à Heroku (le serveur HTTP Gunicorn) quelle version de Python utiliser <br>
`Procfile` indique à Heroku quel type de processus va s'exécuter (processus Web Gunicorn) et le point d'entrée de l'application Python (app.py) <br>
`/assets` contient le CSS du dash <br>
`/models` contient le modele de prediction des match et le code de selection du meilleur modele <br>
`/datas` contient tous les fichiers CSV utils pour les graphes et les predictions <br>

