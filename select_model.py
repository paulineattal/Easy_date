# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:12:28 2022

@author: pauli
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from sklearn.metrics import mean_squared_error, recall_score, f1_score, precision_score


#dir_save le meme que dans le fichier simu_acquisition
data = pd.read_csv("./clean.csv", sep=",")



#ne pas mettre de variables corrélées pour la regression linéaire
X = data.copy().drop(columns=["Unnamed: 0", "match"]) # var descriptives
y = data["match"] # var cible

#si je garde toutes les variables, 
#mais faut voir si le modèle accepte que nos variables soit correlées ou non.


modeles_list = [
    CatBoostClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        verbose=False
    ),
    CatBoostClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=3,
        verbose=False
    )
]


def graphe_pred(ax_pred, yTest, XTest, pred, test_index):
    ypred = pd.DataFrame(pred, columns = ['pred'], index=X.iloc[test_index].index)
    dfpred = pd.concat([yTest, XTest], axis=1)
    dfpred = pd.concat([dfpred, ypred], axis=1)
    dfpred.plot(x='age', y='match', kind='scatter', ax=ax_pred)
    dfpred.plot(x='age', y='pred', kind='scatter', ax=ax_pred, color="r")
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
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    df = pd.DataFrame(columns = ['score', 'rmse', 'ft_imp', 'f1', 'recall', 'precision'])
    for i, modele in enumerate(modeles):
        
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(12, 4))

        
        scores = []
        rmse = []
        ft = []
        f1 = []
        recall=[]
        precision=[]
        df_ft = pd.DataFrame(columns = X.columns.to_list())

        
        for train_index, test_index in kf.split(X,y):
            XTrain, XTest = X.iloc[train_index], X.iloc[test_index]
            yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

            #instanciation, entrainement, test
            model = modele
            model.fit(XTrain, yTrain)
            pred = model.predict(XTest)

            #indicateurs
            #accuracy_score
            scores.append(model.score(XTest, yTest))
            rmse.append(np.sqrt(mean_squared_error(yTest, pred)))
            #travailler sur le macro !!!!
            #bien identifier un match et bien identifier un non match !!!!
            f1.append(f1_score(yTest, pred, average="macro"))#melange rappel et precision 
            recall.append(recall_score(yTest, pred, average="macro")) #(ligne) combien de "positif" j'ai bien predit
            precision.append(precision_score(yTest, pred, average="macro")) #(colonne) 
            df_ft.loc[len(df_ft)] = model.feature_importances_
            
            #afficher chaque pred de chaque découpage de la CV dans le canva        
            graphe_pred(ax_pred, yTest, XTest, pred, test_index)
        
        for t in X.columns.to_list() :
            ft.append(df_ft[t].mean())
        df.loc[i]=[sum(scores)/len(scores), sum(rmse)/len(rmse), ft, sum(f1)/len(f1), sum(recall)/len(recall), sum(precision)/len(precision)]
        
    #afficher l'importances des variables
    graphe_features_importance(df)
    
    return df
    
df_ind = select_model(modeles_list, X, y)
print(df_ind)

#gridSearchCV !!!(methode, dict_patram, score(make_scorer),verbose=,cv=5)
#make_scorer
#np.arange(start=,stop=,step=) pour generaliser ....
#modele.best_estimator_

#courbe ROC et AIC comme metric pour comparer les modeles...

#il faut aussi comprendre et expliquer rapidement le modele


#surechantillonage : methode smote(imblearn.over_sampling) pour les echantillon desequilibré
#pour eviter de predire la classe majoritaire 
#creer des nouveaux point de la classe minoritaire pour equilibrer l'echantillon 
#mais attention au sous-apprentissage !!





