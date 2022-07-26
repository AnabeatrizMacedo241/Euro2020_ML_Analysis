import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import mplsoccer
from soccerplots.radar_chart import Radar
from soccerplots.utils import add_image
import math
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import time
class Euro2020:
    def __init__(self):
        df = pd.read_csv('fbref.csv',sep=';')
        df_GK = pd.read_csv('GK_fbref.csv',sep=';')
        self.df=df
        self.df_GK=df_GK
    def zptile(self,z_score):
        return 0.5 * (math.erf(z_score / 2 ** 0.5) + 1)


class Stats(Euro2020): 
    def SimilarOffense(self, player_name):
        self.df[(self.df['Position'] == 'FW') | (self.df['Position'] =='MF')]
        meanXG = self.df['xG'].mean()
        stdxG = self.df['xG'].std()
        self.df['xG_z'] = (self.df['xG'] - meanXG)/stdxG
        self.df['xG_percent'] = self.df['xG_z'].apply(self.zptile)
        meanXA = self.df['xA'].mean()
        stdxA = self.df['xA'].std()
        self.df['xA_z'] = (self.df['xG'] - meanXA)/stdxA
        self.df['xA_percent'] = self.df['xA_z'].apply(self.zptile)
        meanSoT = self.df['SoT'].mean()
        stdSoT = self.df['SoT'].std()
        self.df['SoT_z'] = (self.df['SoT'] - meanSoT)/stdSoT
        self.df['SoT_percent'] = self.df['SoT_z'].apply(self.zptile)
        meanMin = self.df['Min'].mean()
        stdMin = self.df['Min'].std()
        self.df['Min_z'] = (self.df['Min'] - meanMin)/stdMin
        self.df['Min_percent'] = self.df['Min_z'].apply(self.zptile)
        self.df['Grade'] = (0.4*self.df['xG_percent']+0.3*self.df['xA_percent']+0.2*self.df['SoT_percent']+0.1*self.df['Min_percent'])*10
        self.df.drop("Salary", axis=1, inplace=True)
        model = KMeans()
        model = KMeans(n_clusters=4, random_state=1)
        X = self.df._get_numeric_data()
        y = model.fit_predict(X)
        labels = model.labels_
        self.df["Cluster"] = y
        model.fit(X)
        labels = model.labels_
        pca = PCA(2)
        plot_Cols = pca.fit_transform(X)
        x1 = X.loc[self.df['Player']==player_name]
        list1 = x1.values.tolist()
        cluster1 = model.predict(list1)
        num = int(cluster1)
        players=self.df[(self.df['Cluster'] == num)]
        print(self.df['Cluster'].value_counts())
        players = players.sort_values(by=['Grade'],ascending=False)
        players = players[['Player', 'Position', 'Age', 'Country', 'Grade']]
        
        plt.figure(figsize=(7,4))
        plt.scatter(x=plot_Cols[:,0],y=plot_Cols[:,1],c=labels)
        plt.show()
        if len(player_name)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
        return players.head(15)
    
    
    def SimilarDefense(self,player_name):
        self.df = self.df[(self.df['Position'] == 'DF') | (self.df['Position'] =='MF')]
        meanTklW = self.df['TklW'].mean()
        stdTklW = self.df['TklW'].std()
        self.df['TklW_z'] = (self.df['TklW'] - meanTklW)/stdTklW
        self.df['TklW_percent'] = self.df['TklW_z'].apply(self.zptile)
        meanPast = self.df['Past'].mean()
        stdPast = self.df['Past'].std()
        self.df['Past_z'] = (self.df['Past'] - meanPast)/stdPast
        self.df['Past_percent'] = self.df['Past_z'].apply(self.zptile)
        meanPass = self.df['Pass'].mean()
        stdPass = self.df['Pass'].std()
        self.df['Pass_z'] = (self.df['Pass'] - meanPass)/stdPass
        self.df['Pass_percent'] = self.df['Pass_z'].apply(self.zptile)
        meanMin = self.df['Min'].mean()
        stdMin = self.df['Min'].std()
        self.df['Min_z'] = (self.df['Min'] - meanMin)/stdMin
        self.df['Min_percent'] = self.df['Min_z'].apply(self.zptile)
        meanPress = self.df['Press.1'].mean()
        stdPress = self.df['Press.1'].std()
        self.df['Press_z'] = (self.df['Press.1'] - meanPress)/stdPress
        self.df['Press_percent'] = self.df['Press_z'].apply(self.zptile)
        meanInt = self.df['Int'].mean()
        stdInt = self.df['Int'].std()
        self.df['Int_z'] = (self.df['Int'] - meanInt)/stdInt
        self.df['Int_percent'] = self.df['Int_z'].apply(self.zptile)
        
        self.df['Grade'] = (0.3*self.df['Int_percent']+0.2*self.df['TklW_percent']+0.3*self.df['Press_percent']+0.2*self.df['Pass_percent']+(-0.15*self.df['Past'])+0.1*self.df['Min_percent'])*10
        self.df.drop("Salary", axis=1, inplace=True)
        model = KMeans()
        model = KMeans(n_clusters=4, random_state=1)
        X = self.df._get_numeric_data()
        y = model.fit_predict(X)
        labels = model.labels_
        self.df["Cluster"] = y
        model.fit(X)
        labels = model.labels_
        pca = PCA(2)
        plot_Cols = pca.fit_transform(X)
        x1 = X.loc[self.df['Player']==player_name]
        list1 = x1.values.tolist()
        cluster1 = model.predict(list1)
        num = int(cluster1)
        players=self.df[(self.df['Cluster'] == num)]
        print(self.df['Cluster'].value_counts())
        players = players.sort_values(by=['Grade'],ascending=False)
        players = players[['Player', 'Position', 'Age', 'Country', 'Grade']]
        if len(player_name)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
        return players.head(15)
    
    def RadarChartOff(self,player1, player2):
        radarOff = self.df[['Player','Gls','Sh','SoT','xA','Ast','Sh/90','SoT/90','G/Sh','G/SoT','Dist','xG','npxG','npxG/Sh','G-xG','np:G-xG']]
        players =self.df[(self.df['Player'] == player1) | (self.df['Player'] ==player2)]
        salary = players.Salary.to_list()
        name = players.Player.to_list()
        age = players.Age.to_list()
        
        params = list(radarOff.columns)
        params = params[1:]
    
        ranges = []
        a_values = []
        b_values=[]
    
        for x in params:
            a = min(radarOff[params][x])
            a = a-(a*.25)
            b = max(radarOff[params][x])
            b = b+(b*.25)
            ranges.append((a,b))
    
        for x in range(len(radarOff['Player'])):
            if radarOff['Player'][x] == player1:
                a_values = radarOff.iloc[x].values.tolist()
            if radarOff['Player'][x] == player2:
                b_values = radarOff.iloc[x].values.tolist()
    
        a_values = a_values[1:]
        b_values = b_values[1:]
        values = [a_values,b_values]
    
        title = dict(
            title_name=f'{name[0]}\nAge: {age[0]}',
            title_color = 'red',
            subtitle_name = f'Salary: €{salary[0]}',
            subtitle_color = 'black',
            title_name_2=f'{name[1]}\nAge: {age[1]}',
            title_color_2 = 'blue',
            subtitle_name_2 = f'Salary: €{salary[1]}',
            subtitle_color_2 = 'black',
            title_fontsize=18,
            subtitle_fontsize=15
        )
        endnote = 'Created by Ana Beatriz Macedo/@bibismacedo_14. Data Source from Fbref.com'
        radar = Radar()
    
        fig,ax = radar.plot_radar(ranges=ranges,params=params,values=values,
                                 radar_color=['red','blue'],
                                 alphas=[.75,.6],title=title,endnote=endnote,end_size=9,end_color="black",
                                 compare=True,filename="test.jpg")
        self.df = self.df[(self.df['Position'] == 'FW') | (self.df['Position'] =='MF')]
        meanXG = self.df['xG'].mean()
        stdxG = self.df['xG'].std()
        self.df['xG_z'] = (self.df['xG'] - meanXG)/stdxG
        self.df['xG_percent'] = self.df['xG_z'].apply(self.zptile)
        meanXA = self.df['xA'].mean()
        stdxA = self.df['xA'].std()
        self.df['xA_z'] = (self.df['xG'] - meanXA)/stdxA
        self.df['xA_percent'] = self.df['xA_z'].apply(self.zptile)
        meanSoT = self.df['SoT'].mean()
        stdSoT = self.df['SoT'].std()
        self.df['SoT_z'] = (self.df['SoT'] - meanSoT)/stdSoT
        self.df['SoT_percent'] = self.df['SoT_z'].apply(self.zptile)
        meanMin = self.df['Min'].mean()
        stdMin = self.df['Min'].std()
        self.df['Min_z'] = (self.df['Min'] - meanMin)/stdMin
        self.df['Min_percent'] = self.df['Min_z'].apply(self.zptile)
        self.df['Grade'] = (0.4*self.df['xG_percent']+0.3*self.df['xA_percent']+0.2*self.df['SoT_percent']+0.1*self.df['Min_percent'])*10
        
        plt.figure(figsize=(16,10))
        sns.scatterplot(data=self.df,x='Salary',y='Grade',hue='Position',size='Salary',sizes=(20, 700))
        plt.title('Forwards & Midfielders Salary X Grade',size=20)
        plt.xlabel('Salary',size=20) 
        plt.ylabel('Grade',size=20)
        plt.text(self.df.Salary[self.df.Player==player1],self.df.Grade[self.df.Player==player1],player1, color='red',fontdict=dict(color='red', alpha=1, size=16))
        plt.text(self.df.Salary[self.df.Player==player2],self.df.Grade[self.df.Player==player2],player2,color='blue',fontdict=dict(color='black', alpha=1, size=16))
        sns.despine(bottom = True, left = True)
        sns.set(rc={'axes.facecolor':'lightgreen', 'figure.facecolor':'lightgreen'})
        if len(player1)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
        if len(player2)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
       
    
    def RadarChartDef(self,player1, player2):
        radarDef = self.df[['Player','TklW','Tkl%','Succ%','Past','Pass','Press.1','Blocks2','Int','TklInt','Clr2','Min']]
        players =self.df[(self.df['Player'] == player1) | (self.df['Player'] ==player2)]
        salary = players.Salary.to_list()
        name = players.Player.to_list()
        age = players.Age.to_list()
    
        params = list(radarDef.columns)
        params = params[1:]
    
        ranges = []
        a_values = []
        b_values=[]
    
        for x in params:
            a = min(radarDef[params][x])
            a = a-(a*.25)
            b = max(radarDef[params][x])
            b = b+(b*.25)
            ranges.append((a,b))
    
        for x in range(len(radarDef['Player'])):
            if radarDef['Player'][x] == player1:
                a_values = radarDef.iloc[x].values.tolist()
            if radarDef['Player'][x] == player2:
                b_values = radarDef.iloc[x].values.tolist()
    
        a_values = a_values[1:]
        b_values = b_values[1:]
        values = [a_values,b_values]
    
        title = dict(
            title_name=f'{name[0]}\nAge: {age[0]}',
            title_color = 'red',
            subtitle_name = f'Salary: €{salary[0]}',
            subtitle_color = 'black',
            title_name_2=f'{name[1]}\nAge: {age[1]}',
            title_color_2 = 'blue',
            subtitle_name_2 = f'Salary: €{salary[1]}',
            subtitle_color_2 = 'black',
            title_fontsize=18,
            subtitle_fontsize=15
        )
        endnote = 'Created by Ana Beatriz Macedo/@bibismacedo_14. Data Source from Fbref.com'
        radar = Radar()
    
        fig,ax = radar.plot_radar(ranges=ranges,params=params,values=values,
                                 radar_color=['red','blue'],
                                 alphas=[.75,.6],title=title,endnote=endnote,end_size=9,end_color="black",
                                 compare=True,filename="test.jpg")
        
        self.df = self.df[(self.df['Position'] == 'DF') | (self.df['Position'] =='MF')]
        meanTklW = self.df['TklW'].mean()
        stdTklW = self.df['TklW'].std()
        self.df['TklW_z'] = (self.df['TklW'] - meanTklW)/stdTklW
        self.df['TklW_percent'] = self.df['TklW_z'].apply(self.zptile)
        meanPast = self.df['Past'].mean()
        stdPast = self.df['Past'].std()
        self.df['Past_z'] = (self.df['Past'] - meanPast)/stdPast
        self.df['Past_percent'] = self.df['Past_z'].apply(self.zptile)
        meanPass = self.df['Pass'].mean()
        stdPass = self.df['Pass'].std()
        self.df['Pass_z'] = (self.df['Pass'] - meanPass)/stdPass
        self.df['Pass_percent'] = self.df['Pass_z'].apply(self.zptile)
        meanMin = self.df['Min'].mean()
        stdMin = self.df['Min'].std()
        self.df['Min_z'] = (self.df['Min'] - meanMin)/stdMin
        self.df['Min_percent'] = self.df['Min_z'].apply(self.zptile)
        meanPress = self.df['Press.1'].mean()
        stdPress = self.df['Press.1'].std()
        self.df['Press_z'] = (self.df['Press.1'] - meanPress)/stdPress
        self.df['Press_percent'] = self.df['Press_z'].apply(self.zptile)
        meanInt = self.df['Int'].mean()
        stdInt = self.df['Int'].std()
        self.df['Int_z'] = (self.df['Int'] - meanInt)/stdInt
        self.df['Int_percent'] = self.df['Int_z'].apply(self.zptile)
        
        self.df['Grade'] = (0.3*self.df['Int_percent']+0.2*self.df['TklW_percent']+0.3*self.df['Press_percent']+0.2*self.df['Pass_percent']+(-0.15*self.df['Past'])+0.1*self.df['Min_percent'])*10
       
        plt.figure(figsize=(16,10))
        sns.scatterplot(data=self.df,x='Salary',y='Grade',hue='Position',size='Salary',sizes=(20, 700))
        plt.title('Defenders & Midfielders Salary X Grade',size=20)
        plt.xlabel('Salary',size=20) 
        plt.ylabel('Grade',size=20)
        plt.text(self.df.Salary[self.df.Player==player1],self.df.Grade[self.df.Player==player1],player1, color='red',fontdict=dict(color='red', alpha=1, size=16))
        plt.text(self.df.Salary[self.df.Player==player2],self.df.Grade[self.df.Player==player2],player2,color='blue',fontdict=dict(color='black', alpha=1, size=16))
        sns.despine(bottom = True, left = True)
        sns.set(rc={'axes.facecolor':'lightgreen', 'figure.facecolor':'lightgreen'})
        if len(player1)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
        if len(player2)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
    def RadarChartGK(self,player1, player2):
        radarGK = self.df_GK[['Player','GA','GA90','90s','SoTA','Saves','Save%','CS','PKatt','PKA','PKsv']]
        players =self.df_GK[(self.df_GK['Player'] == player1) | (self.df_GK['Player'] ==player2)]
        salary = players.Salary.to_list()
        name = players.Player.to_list()
        age = players.Age.to_list()
        
        params = list(radarGK.columns)
        params = params[1:]
    
        ranges = []
        a_values = []
        b_values=[]
    
        for x in params:
            a = min(radarGK[params][x])
            a = a-(a*.25)
            b = max(radarGK[params][x])
            b = b+(b*.25)
            ranges.append((a,b))
    
        for x in range(len(radarGK['Player'])):
            if radarGK['Player'][x] == player1:
                a_values = radarGK.iloc[x].values.tolist()
            if radarGK['Player'][x] == player2:
                b_values = radarGK.iloc[x].values.tolist()
    
        a_values = a_values[1:]
        b_values = b_values[1:]
        values = [a_values,b_values]
    
        title = dict(
            title_name= f'{name[0]}\nAge: {age[0]}',
            title_color = 'red',
            subtitle_name = f'Salary: €{salary[0]}',
            subtitle_color = 'black',
            title_name_2=f'{name[1]}\nAge: {age[1]}',
            title_color_2 = 'blue',
            subtitle_name_2 = f'Salary: €{salary[1]}',
            subtitle_color_2 = 'black',
            title_fontsize=18,
            subtitle_fontsize=15
        )
        endnote = 'Created by Ana Beatriz Macedo/@bibismacedo_14. Data Source from Fbref.com'
        radar = Radar()
    
        fig,ax = radar.plot_radar(ranges=ranges,params=params,values=values,
                                 radar_color=['red','blue'],
                                 alphas=[.75,.6],title=title,endnote=endnote,end_size=9,end_color="black",
                                 compare=True,filename="test.jpg")
        
        meanSaves = self.df_GK['Saves'].mean()
        stdSaves = self.df_GK['Saves'].std()
        self.df_GK['Saves_z'] = (self.df_GK['Saves'] - meanSaves)/stdSaves
        self.df_GK['Saves_percent'] = self.df_GK['Saves_z'].apply(self.zptile)
        meanCS = self.df_GK['CS'].mean()
        stdCS = self.df_GK['CS'].std()
        self.df_GK['CS_z'] = (self.df_GK['CS'] - meanCS)/stdCS
        self.df_GK['CS_percent'] = self.df_GK['CS_z'].apply(self.zptile)
        meanGA = self.df_GK['GA'].mean()
        stdGA = self.df_GK['GA'].std()
        self.df_GK['GA_z'] = (self.df_GK['GA'] - meanGA)/stdGA
        self.df_GK['GA_percent'] = self.df_GK['GA_z'].apply(self.zptile)
        meanPKsv = self.df_GK['PKsv'].mean()
        stdPKsv = self.df_GK['PKsv'].std()
        self.df_GK['PKsv_z'] = (self.df_GK['PKsv'] - meanPKsv)/stdPKsv
        self.df_GK['PKsv_percent'] = self.df_GK['PKsv_z'].apply(self.zptile)
        self.df_GK['Grade'] = (0.4*self.df_GK['Saves_percent']+0.3*self.df_GK['CS_percent']+(-0.2*self.df_GK['GA_percent'])+0.3*self.df_GK['PKsv_percent'])*10
       
        plt.figure(figsize=(16,10))
        sns.scatterplot(data=self.df_GK,x='Salary',y='Grade',size='Salary',sizes=(20, 700))
        plt.title('Goalkeepers Salary X Grade',size=20)
        plt.xlabel('Salary',size=20) 
        plt.ylabel('Grade',size=20)
        plt.text(self.df_GK.Salary[self.df_GK.Player==player1],self.df_GK.Grade[self.df_GK.Player==player1],player1, color='red',fontdict=dict(color='red', alpha=1, size=16))
        plt.text(self.df_GK.Salary[self.df_GK.Player==player2],self.df_GK.Grade[self.df_GK.Player==player2],player2,color='blue',fontdict=dict(color='black', alpha=1, size=16))
        sns.despine(bottom = True, left = True)
        sns.set(rc={'axes.facecolor':'lightgreen', 'figure.facecolor':'lightgreen'})
        
        players = self.df_GK[['Player', 'Age', 'Grade', 'Salary']]
        players = players.sort_values(by=['Grade'],ascending=False)
        if len(player1)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
        if len(player2)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')    
       
        return players.head(10)
    
    def Scorers(self):
        scorer = self.df[['Player','Country','Gls','xG','SoT%']]
        scorer = scorer.sort_values(by=['Gls'],ascending=False)
        return scorer.head(15)
    
    def Passers(self):
        passer = self.df[['Player','Country','Ast','xA','Cmp%']]
        passer = passer.sort_values(by=['Ast'],ascending=False)
        return passer.head(15)
    
    def get_player(self, player_name):
        player=self.df[(self.df['Player'] == player_name)]
        if len(player)==0:
            raise  ExceptionPlayerName('No results for this player, check if you wrote his name correctly.')
        return(player)
    
class ExceptionPlayerName(Exception):
    pass