# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:12:28 2022

@author: pauli
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, std
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.metrics import mean_squared_error


#dir_save le meme que dans le fichier simu_acquisition
dir_save = "/home/cognidis/Documents/stage_2/data_pred/"
by_stop= pd.read_csv(dir_save + "data_init.csv")

by_stop["avg_arrival"] = by_stop["avg_arrival"].fillna(0)
by_stop["avg_start"] = by_stop["avg_start"].fillna(0)


#ne pas mettre de variables corrélées pour la regression linéaire
by_stop = by_stop[['avg_arrival', 'stop_id', 'freq_by_stop', 'Ind_by_intersect', 'size_ent_by_stop']].set_index('stop_id')
X = by_stop[by_stop.columns[1:]] # var descriptives
y = by_stop["avg_arrival"] # var cible

#si je garde toutes les variables, 
#mais faut voir si le modèle accepte que nos variables soit correlées ou non.
#X = by_stop[by_stop.columns[1:-2]] # var descriptives
#y = by_stop["avg_arrival"] # var cible


modeles_list = [
    CatBoostRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        verbose=False,
        objective='Poisson'
    ),
    CatBoostRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=3,
        verbose=False,
        objective='Poisson'
    )
]


def graphe_pred(ax_pred, yTest, XTest, pred, test_index):
    ypred = pd.DataFrame(pred, columns = ['pred'], index=X.iloc[test_index].index)
    dfpred = pd.concat([yTest, XTest], axis=1)
    dfpred = pd.concat([dfpred, ypred], axis=1)
    dfpred.plot(x='size_ent_by_stop', y='avg_arrival', kind='scatter', ax=ax_pred)
    dfpred.plot(x='size_ent_by_stop', y='pred', kind='scatter', ax=ax_pred, color="r")
    plt.xlabel("effectif entreprise", size = 12)
    plt.ylabel("nombre d\'OD", size = 12)
    

def graphe_features_importance(df_ind):
    for i, ft in enumerate(df_ind["ft_imp"]) : 
        imp = {"VarName":X.columns,"Importance": ft}
        tmpImp = pd.DataFrame(imp).sort_values(by="Importance",ascending=False)
        print("configuration n°" + str(i))
        plt.figure(figsize=(3, 3))
        sns.barplot(orient='h',data=tmpImp,y='VarName',x='Importance',color='silver')
        plt.show()

def select_model(modeles, X, y) :
    kf = KFold(n_splits=5, shuffle=True)
    df = pd.DataFrame(columns = ['score', 'rmse', 'ft_imp'])
    for i, modele in enumerate(modeles):
        
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(12, 4))

        scores = []
        rmse = []
        ft = []
        df_ft = pd.DataFrame(columns = X.columns.to_list())

        for train_index, test_index in kf.split(X):
            XTrain, XTest = X.iloc[train_index], X.iloc[test_index]
            yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

            #instanciation, entrainement, test
            model = modele
            model.fit(XTrain, yTrain)
            pred = model.predict(XTest)

            #indicateurs
            scores.append(model.score(XTest, yTest))
            rmse.append(np.sqrt(mean_squared_error(yTest, pred)))
            df_ft.loc[len(df_ft)] = model.feature_importances_
            
            #afficher chaque pred de chaque découpage de la CV dans le canva        
            graphe_pred(ax_pred, yTest, XTest, pred, test_index)
        
        for t in X.columns.to_list() :
            ft.append(df_ft[t].mean())
        df.loc[i]=[sum(scores)/len(scores), sum(rmse)/len(rmse), ft]
        
    #afficher l'importances des variables
    graphe_features_importance(df)
    
    return df
    
df_ind = select_model(modeles_list, X, y)