# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:12:28 2022

@author: pauli
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, recall_score, f1_score, make_scorer,precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("./trainClean.csv", sep=",")
X_sub = pd.read_csv("./submissionsClean.csv", sep=',')
iid_pid = X_sub['iid_pid']
X_sub.drop(columns=["idg","partner","duo","iid_pid", "iid", "id", "pid","zipcode"])
X = data.copy().drop(columns=["idg","partner","duo","iid_pid", "iid", "id", "pid","zipcode",  "match","wave","round","position","positin1", "order", "expnum", "dec_o"]) # var descriptives
y = data["match"] # var cible

#np.arange(start = 5, stop = 250, step = 50)
params_modeles = [
              # {'estimator__max_features' : [5,10,20],
              # 'estimator__min_samples_split' : [50],
              # 'estimator__max_depth':[1000,30],
              # 'estimator__min_samples_leaf':[10]
              # }
              # {"estimator__solver":["newton-cg"],
              #   "estimator__penalty":["none"],
              #   "estimator__max_iter":[10000],
              #   "estimator__warm_start":[False]},
              # {"estimator__n_estimators":[1000],
              #   "estimator__criterion":["gini"],
              #   "estimator__max_depth":[1000],
              #   "estimator__min_samples_split":[50],
              #   "estimator__min_samples_leaf":[50]},
                 {"estimator__loss":["log_loss"],
                  "estimator__learning_rate":[0.1,0.2,0.5,0.7,1],
                  "estimator__n_estimators":[6000],
                  "estimator__subsample":[1],
                  "estimator__criterion":["friedman_mse"],
                  "estimator__max_depth":[2],
                  "estimator__min_samples_split":[20],
                  "estimator__min_samples_leaf":[10]},
               # {'random_state':[1], 
               #  "max_iter":[100]},
               # {"kernel":["linear"]}
              # {"estimator__C": [1,2,4,8],
              # "estimator__kernel": ["poly","rbf"],
              # "estimator__degree":[1, 2, 3, 4]},
              # 
            #   {"MLPRegressor__solver": ["bfgs"],
            #   "MLPRegressor__max_iter": [100,200,300,500],
            #   "MLPRegressor__activation" : ['relu','logistic','tanh'],
            #   "MLPRegressor__hidden_layer_sizes":[(2,), (4,),(2,2),(4,4),(4,2),(10,10),(2,2,2)]}
              ]
modeles_list = [
                # DecisionTreeClassifier()
                # LogisticRegression(),
                # RandomForestClassifier(),
                GradientBoostingClassifier(),
                # MLPClassifier(),
                # SVC()
                # OneVsRestClassifier(SVC())
                # Pipeline([('scaler', StandardScaler()),('MLPRegressor', MLPRegressor())])
                ]



def select_model(modeles, parameters, X, y) :
    df = pd.DataFrame(columns = ['best','score', 'ft_imp','rmse', 'f1', 'recall', 'precision'])
    XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size = 0.30, stratify = y)
    for i, modele in enumerate(modeles):
        
        f1 = make_scorer(f1_score , average='macro')
        from sklearn.model_selection import StratifiedKFold
        sel = RFECV(estimator=modele, step=1, cv=StratifiedKFold(5),scoring = f1)

        k_fold = StratifiedKFold(n_splits=5, shuffle=True)
        model = GridSearchCV(estimator=sel,
                            param_grid=parameters[i],
                            scoring = f1,
                            verbose = False,
                            cv = k_fold)
        

        model.fit(XTrain, yTrain)
        
        print("--------------------")
        print(model.best_score_)
        print(model.best_estimator_)
        print(model.best_params_)
        features=list(X.columns[model.best_estimator_.support_])
        print(features)
        
        pred = model.predict(XTest)
        rmse_ = np.sqrt(mean_squared_error(yTest, pred))
        f1_score_ = f1_score(yTest, pred, average="macro")#melange rappel et precision 
        recall_ = recall_score(yTest, pred, average="macro") #(ligne) combien de "positif" j'ai bien predit
        precision_ = precision_score(yTest, pred, average="macro", zero_division=1) #(colonne) 

        df.loc[i]=[model.best_estimator_, model.best_score_, features, rmse_, f1_score_, recall_, precision_]
        
    return df
    
df_ind = select_model(modeles_list, params_modeles, X, y)

#keep model with best f1 score
indice=df_ind['f1'].idxmax()
best_model = df_ind["best"][indice]
#keep only features selctioned
X_sub = X_sub[df_ind["ft_imp"][indice]]
X = X[df_ind["ft_imp"][indice]]

best_model.fit(X, y)
pred = best_model.predict(X_sub)

df_pred = pd.DataFrame(iid_pid, columns = ["iid_pid"]).assign(target=pred)
df_pred.iid_pid = df_pred.iid_pid.astype(int)
df_pred.target = df_pred.target.astype(int)
df_pred.to_csv("./submissionPred.csv", index=False)


#courbe ROC et AIC comme metric pour comparer les modeles...
#il faut aussi comprendre et expliquer rapidement le modele



XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size = 0.30, stratify = y)

best_model = GradientBoostingClassifier(loss="log_loss", learning_rate=0.5, n_estimators=6000, subsample=1, criterion="friedman_mse", max_depth=2, min_samples_split=20, min_samples_leaf=10)
best_model.fit(XTrain, yTrain)
pred = best_model.predict(XTest)
f1_score_ = f1_score(yTest, pred, average="macro")
print(f1_score_)



#pred = (numpy.random.choice([0,1], len(X_sub), replace = True, p = [len(data[data["match"]==0.0])/len(data), len(data[data["match"]==1.0])/len(data)]))
#DecisionTreeClassifier(max_depth=1000, min_samples_leaf=10,min_samples_split=10)
#LogisticRegression(max_iter=10000,penalty='none', solver='newton-cg')
#random forest pas fou...
#DecisionTreeClassifier(max_depth=1000, min_samples_leaf=10,min_samples_split=10)   

#gradient boosting avec 5000 abres et LR a 0.5 parrait correct a verifier
#best_model = GradientBoostingClassifier(loss="log_loss", learning_rate=0.4, n_estimators=6000, subsample=1, criterion="friedman_mse", max_depth=2, min_samples_split=20, min_samples_leaf=10)
    #6000 et LR 0.2 ==> 0.662333650313972
    #6000 et 0.4 ==>0."6695555235903337
    #6000 et 0.5 ==> 0.664058503458411






