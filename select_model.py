# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:12:28 2022

@author: pauli
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, recall_score, f1_score, make_scorer,precision_score


#dir_save le meme que dans le fichier simu_acquisition
data = pd.read_csv("./trainClean.csv", sep=",")



#ne pas mettre de variables corrélées pour la regression linéaire
X = data.copy().drop(columns=["match"]) # var descriptives
y = data["match"] # var cible

#si je garde toutes les variables, 
#mais faut voir si le modèle accepte que nos variables soit correlées ou non.


parameters = [{'max_depth' : np.arange(start = 1, stop = 10, step = 1) , 
              'min_samples_leaf' : np.arange(start = 5, stop = 50, step = 50),
              'min_samples_split' : np.arange(start = 10, stop = 100, step = 50)}]

modeles_list = [DecisionTreeClassifier()]


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
    df = pd.DataFrame(columns = ['best','score', 'rmse', 'ft_imp', 'f1', 'recall', 'precision'])
    XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size = 0.30, stratify = y)
    for i, modele in enumerate(modeles):
        
        #fig_pred, ax_pred = plt.subplots(1, 1, figsize=(12, 4))
    
        f1 = make_scorer(f1_score , average='macro')
        model = GridSearchCV(modele,
                            parameters[i],
                            scoring = f1,
                            verbose = False,
                            cv = 5)

        model.fit(XTrain, yTrain)
        pred = model.predict(XTest)
        
        #indicateurs
        #accuracy_score
        score_ = model.score(XTest, yTest)
        rmse_ = np.sqrt(mean_squared_error(yTest, pred))
        f1_score_ = f1_score(yTest, pred, average="macro")#melange rappel et precision 
        recall_ = recall_score(yTest, pred, average="macro") #(ligne) combien de "positif" j'ai bien predit
        precision_ = precision_score(yTest, pred, average="macro", zero_division=1) #(colonne) 

        #afficher chaque pred de chaque découpage de la CV dans le canva        
        #graphe_pred(ax_pred, yTest, XTest, pred, test_index)
        
        df.loc[i]=[model.best_estimator_, score_, rmse_, model.best_estimator_.feature_importances_, f1_score_, recall_, precision_]
        
    #afficher l'importances des variables
    #graphe_features_importance(df)
    
    return df
    
df_ind = select_model(modeles_list, X, y)
print(df_ind)



#courbe ROC et AIC comme metric pour comparer les modeles...

#il faut aussi comprendre et expliquer rapidement le modele


