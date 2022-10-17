import numpy as np

class PrepDatas:
    
    def __init__(self, df):
        self.__df = df
    
    def get_df(self):
        return self.__df
    
    def get_df_men(self):
        return self.__df[self.__df["gender"]=="men"]
        
    def get_df_women(self):
        return self.__df[self.__df["gender"]=="women"]
    
    def get_df_boxplot(self):
        attrs_box=["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
        return self.__df[attrs_box].rename(columns={'attr1_1':'Attirant', 'sinc1_1':'Sincere','intel1_1':'Intelligent','fun1_1': 'Fun','amb1_1':'Ambitieux', 'shar1_1':'Interets Communs'})
    
    def build_df_graphes(self):
        df=self.__df

        #group goals by categories
        conditions_goal = [
            (df['goal'] == 1) | (df['goal'] == 2),
            (df['goal'] == 4) | (df['goal'] == 3),
            (df['goal'] == 5) | (df['goal'] == 6)
            ]
        values_goal = ['fun', 'serious', 'pass time']
        df['goal_cat'] = np.select(conditions_goal, values_goal)
        
        #group ages by categories
        conditions_age = [
            (df['age'] > min(df["age"])) & (df['age'] <= 22),
            (df['age'] > 22) & (df['age'] <= 30),
            (df['age'] > 30)
            ]
        values_age = ['young', 'adult', 'old']
        df['age_cat'] = np.select(conditions_age, values_age)
        
        
        df["culture_interest"]=np.where(df['culture']>=6, 'yes_culture', 'no_culture')
        df["indoors_interest"]=np.where(df['indoors']>=6, 'yes_indoors', 'no_indoors')
        df["sport_interest"]=np.where(df['sport']>=6, 'yes_sport', 'no_sport')
        df["social_interest"]=np.where(df['social']>=6, 'yes_social', 'no_social')
        df["most_interest"]=df[["culture","indoors","sport","social"]].idxmax(axis=1)
        
        #map genders
        df["gender"]=df["gender"].replace(0, "women")
        df["gender"]=df["gender"].replace(1, "men")
        
        self.__df=df
        
    
        
        
    
