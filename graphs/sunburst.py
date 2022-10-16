
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("./datas/trainGraph.csv", sep=",")

conditions = [
    (df['goal'] == 1) | (df['goal'] == 2),
    (df['goal'] == 4) | (df['goal'] == 3),
    (df['goal'] == 5) | (df['goal'] == 6)
    ]

values = ['fun', 'serious', 'pass time']

df['goal_cat'] = np.select(conditions, values)

conditions = [
    (df['age'] > min(df["age"])) & (df['age'] <= 22),
    (df['age'] > 22) & (df['age'] <= 30),
    (df['age'] > 30)
    ]

values = ['young', 'adult', 'old']

df['age_cat'] = np.select(conditions, values)

df["culture_interest"]=np.where(df['culture']>=6, 'yes_culture', 'no_culture')
df["indoors_interest"]=np.where(df['indoors']>=6, 'yes_indoors', 'no_indoors')
df["sport_interest"]=np.where(df['sport']>=6, 'yes_sport', 'no_sport')
df["social_interest"]=np.where(df['social']>=6, 'yes_social', 'no_social')

df["most_interest"]=df[["culture","indoors","sport","social"]].idxmax(axis=1)

df["gender"]=df["gender"].replace(0, "women")
df["gender"]=df["gender"].replace(1, "men")


df_men = df[df["gender"]=="men"]
df_women = df[df["gender"]=="women"]

fig = px.sunburst(df_men, path=['most_interest', 'goal_cat', 'age_cat'],
                  values='income', color='income'
                 )
fig.show()

fig = px.sunburst(df_women, path=['most_interest', 'goal_cat', 'age_cat'],
                  values='income', color='income'
                 )
fig.show()