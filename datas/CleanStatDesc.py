#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/M2/Projet Easy Date Python M2 SISE/train.csv", sep=";")
df.drop_duplicates()
pd.set_option("display.max_columns", 70)
pd.set_option('display.max_rows', 70)
df.head()


# In[ ]:


#Changer les virgules en points
colsToReplace = ["int_corr",'attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1',"pf_o_att","pf_o_sin","pf_o_int","pf_o_fun","pf_o_amb","pf_o_sha"]
df[colsToReplace] = df[colsToReplace].replace(",", ".", regex=True)
df[["zipcode","income"]] = df[["zipcode","income"]].replace(",", "", regex=True)

#Changer les str en float
colsToFloat = ["attr1_1","sinc1_1","intel1_1","fun1_1","amb1_1","shar1_1","income","int_corr","zipcode","pf_o_att","pf_o_sin","pf_o_int","pf_o_fun","pf_o_amb","pf_o_sha"]
df[colsToFloat] = df[colsToFloat].apply(pd.to_numeric, downcast="float", errors='coerce')


# In[ ]:


df_num = df.select_dtypes(include=["int64","float32","float64"])
intCols = []
for col in df_num.columns:
  if (df[col].fillna(-9999) % 1  == 0).all():
    intCols.append(col)
print(intCols)


# In[ ]:



print(df_num.info())
sum_rates = df[["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]].sum(axis=1)
print(sum_rates>100)
sum_rates.plot.box()
df.int_corr.plot.box()
df.income.plot.box()

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights="distance")
df_new = pd.DataFrame(imputer.fit_transform(df_num), columns = df_num.columns)
df_new[intCols] = round(df_new[intCols])
sum_rates = df[["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]].sum(axis=1)
print(sum_rates>100)
sum_rates.plot.box()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
train = df_new.copy()
pd.set_option('max_colwidth', 1000)
pd.set_option("display.max_columns",70)
train.dtypes


# In[ ]:


train.head()


# In[ ]:


train["nb"]= 1
train = train.dropna(subset=["iid","pid"])
train[["iid","pid"]] = train[["iid","pid"]].apply(pd.to_numeric, downcast="integer", errors='coerce')
train["duo"] = np.where(train["iid"] > train["pid"], train["iid"].astype(str) + "_" + train["pid"].astype(str), train["pid"].astype(str) + "_" + train["iid"].astype(str))
train.to_csv("trainClean.csv", index=False)


# In[ ]:


train[["iid", "wave"]].drop_duplicates()["wave"].value_counts().sort_values(ascending=False)


# In[ ]:


print(train.drop_duplicates("duo")["match"].value_counts())
train.drop_duplicates("duo")["match"].value_counts(normalize=True)


# In[ ]:


print(train.drop_duplicates("iid")["gender"].value_counts())
train.drop_duplicates("iid")["gender"].value_counts(normalize=True)


# In[ ]:


dataQ4 = pd.crosstab(train.order, train.match, train.nb, aggfunc="sum", normalize='index').reset_index()
dataQ4.columns = ["order", "no", "yes"]
print(dataQ4)
from scipy.stats import pearsonr
pearsonr(dataQ4.order, dataQ4.yes)
#Plus l'ordre est grand moins il y a de match, significatif


# In[ ]:


train[["iid","sports","tvsports","exercise","dining","museums","art","hiking","gaming","clubbing",
       "reading","tv","theater","movies","concerts","music","shopping","yoga"]].groupby(["iid"]).mean().mean().sort_values(ascending=False)


# In[ ]:


dataQ7 = train[["iid","attr1_1","sinc1_1","intel1_1","fun1_1","amb1_1","shar1_1"]].groupby("iid").mean()


# In[ ]:


corr = round(dataQ7.corr(),2)
print(corr)
pd.plotting.scatter_matrix(dataQ7, figsize=(6, 6))
plt.show()


# In[ ]:


dataQ8 = train.drop_duplicates("duo")[["int_corr","match"]]
print(dataQ8.corr())
dataQ8["int_corrPred"] = np.where(dataQ8['int_corr']>=0, 1, 0)


# In[ ]:


from sklearn import metrics
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(dataQ8["match"], dataQ8["int_corrPred"])).plot()
plt.show()
Accuracy = metrics.accuracy_score(dataQ8["match"], dataQ8["int_corrPred"])
Precision = metrics.precision_score(dataQ8["match"], dataQ8["int_corrPred"])
Sensitivity_recall = metrics.recall_score(dataQ8["match"], dataQ8["int_corrPred"])
Specificity = metrics.recall_score(dataQ8["match"], dataQ8["int_corrPred"], pos_label=0)
F1_score = metrics.f1_score(dataQ8["match"], dataQ8["int_corrPred"])
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})


# In[ ]:


field_txt = {1 :"Law", 2 : "Math", 3 : "Social Science, Psychologist", 4 : "Medical Science, Pharmaceuticals, and Bio Tech", 5 : "Engineering", 6 : "English/Creative Writing/ Journalism",
7 : "History/Religion/Philosophy", 8 : "Business/Econ/Finance", 9 : "Education, Academia", 10 : "Biological Sciences/Chemistry/Physics", 11 : "Social Work", 12 : "Undergrad/undecided",
13 : "Political Science/International Affairs", 14 : "Film", 15 : "Fine Arts/Arts Administration", 16 : "Languages", 17 : "Architecture", 18 : "Other"}
train.replace({"field_cd" : field_txt}).field_cd.value_counts()


# In[ ]:


#

