
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Fig :

    def __init__(self, df,df_men,df_women,df_boxplot,df_word):
        self.__df = df
        self.__df_men = df_men
        self.__df_women = df_women
        self.__df_boxplot = df_boxplot
        self.__df_word = df_word
        
    def get_fig_sunburst_men(self):
        return px.sunburst(self.__df_men, title="Hommes", path=['most_interest', 'goal_cat', 'age_cat'],values='expnum', color='expnum').update_layout({'plot_bgcolor':'rgb(39, 43, 48)', 'paper_bgcolor':'rgb(39, 43, 48)'})
    
    def get_fig_sunburst_women(self):
        return px.sunburst(self.__df_women, title="Femmes", path=['most_interest', 'goal_cat', 'age_cat'],values='expnum', color='expnum').update_layout({'plot_bgcolor':'rgb(39, 43, 48)', 'paper_bgcolor':'rgb(39, 43, 48)'})

    def get_fig_boxplot(self):
        fig_plot = go.Figure()
        colors = ['rgba(240, 248, 255, 1 )', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)','rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
        for xd, yd, cls in zip(self.__df_boxplot.columns, [self.__df_boxplot[i].to_list() for i in self.__df_boxplot.columns], colors):
                fig_plot.add_trace(go.Box(
                y=yd,
                name=xd,
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2,
                fillcolor=cls,
                marker_size=2
                    )
            )
        fig_plot.update_layout(
            yaxis=dict(
                autorange=True,
                showgrid=True,
                zeroline=True,
                dtick=5,
                gridcolor='rgb(255, 255, 255)',
                gridwidth=1,
                zerolinecolor='rgb(255, 255, 255)',
                zerolinewidth=2,
            ),
            margin=dict(
                l=40,
                r=30,
                b=80,
                t=100,
            ),
            paper_bgcolor='rgb(39, 43, 48)',
            plot_bgcolor='rgb(243, 243, 243)',
            showlegend=False
        )
        return fig_plot
    
    def fig_words(self):    
        wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(self.__df_word)
        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0,0)
        plt.savefig("./assets/word.png", bbox_inches = 'tight', pad_inches = 0)
        

