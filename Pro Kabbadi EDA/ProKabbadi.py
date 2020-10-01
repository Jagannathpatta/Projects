import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os.path
from os import path
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# import plotly.express as px
# https://docs.streamlit.io/en/stable/api.html
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process



@st.cache
def load_data(URL):

    data = pd.read_csv(URL , index_col='Unnamed: 0')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    
    return data


################################### Getting Top players ##################################################

def GetTopScores(cat , N = 5):

    if cat == 'Allrounders':
        col = 'player_total_points'
        lab = 'Players with Most Points in a Match'
    elif cat ==  'Raiders':
        col = 'player_raid_points_total'
        lab = 'Players with Most Raid Points in a Match'
    elif cat ==  'Defenders':
        col = 'player_tackle_points_total'
        lab = 'Players with Most Tackel Points in a Match'

    res = pd.DataFrame( columns = ['player_id','player_name', col] )

    for index in Player_data.groupby(['match_id' ,'player_id'])[col].sum().sort_values(ascending = False).index[:N] :
        df = Player_data[(Player_data.match_id == index[0]) & (Player_data.player_id == index[1])]
        res = pd.concat([ res , df[['player_id','player_name',col]]])
    res.set_index(pd.Series(range(1 ,res.shape[0]+1)) , inplace = True)
        
    return res, lab


def GetTopPlayers(cat , N = 5):
    
    df , lab = GetTopScores(cat , N)
    
    df.drop_duplicates(subset=['player_id'], keep='first' , inplace = True)
    lab = 'List of unique top '+ str(df.shape[0]) +' out of ' +str( N )+ ' '+lab
    
    return df, lab


#######################################-- PLAYER DETAILS --###########################################


def teams_player_played_for(Id):
    
    # Below line gets us the ids of thoes teams the player played for.
    team_ids = Player_data.groupby(['player_id','team_id']).match_id.unique().loc[Id].index
    
    # defining an empty DataFrame that will be returned.
    team_names = pd.DataFrame()

    # defining an empty list to store the no. of matches played by the player for different teams.
    team_match_counts = []

    # List of column names from Player_data Dataset to get player's perfomance stats.
    cols = ['player_total_points','player_raid_points_total','player_tackle_points_total','player_raids_total','player_raids_successful',
            'player_raids_unsuccessful','player_raids_empty','player_tackles_total','player_tackles_successful','player_tackles_unsuccessful'] 
    
    # defining dictionary that will contain the empty list for every col in cols.
    col_dict = {}
    for col in cols:
        col_dict[col] = []

    # Looping through team_ids
    for id in team_ids:
        
        # here groupby() of pandas is used.
        # getting the count i.e number of matches the player with player_id = Id has played for the team with team_id = id.
        # and appending into the team_match_counts list.
        team_match_counts.append(len(Player_data.groupby(['player_id','team_id']).match_id.unique().loc[Id].loc[id]))

        # summing up the points and stats of player with player_id = Id has played for the team with team_id = id.
        # and appending into respective col in cols.
        for col in cols:
            col_dict[col].append(Player_data.groupby(['player_id','team_id'])[col].sum().loc[Id].loc[id])
    
        # adding teams data (like : id, name, short_name, counts) to the empty dataframe.
        team_names = pd.concat([team_names , pd.DataFrame(teams[teams.id == id].values)])
        

    # naming the columns of dataframe.
    team_names.columns = ['team_id','Name','short_name']
    team_names['Matches_Played'] = team_match_counts
    
    # adding Player_data stats into the DataFrame.
    for col in cols:
        team_names[col[7:]] = col_dict[col]

    # returning the DataFrame.
    return team_names.set_index('Name').T


def Get_player_details( Id ): 
  
    # creating an empty dictionary to return results.
    details = {}

    # getting the name of the player from players_catalog.
    details['Name'] = Player_data[Player_data.player_id == Id].player_name.values[0]

    # getting the total no. of matches played by the player in his entire career.
    details['Total No. of matches played'] = len(Player_data[(Player_data.player_id == Id) & (Player_data.player_played == True) ].match_id.unique())

    # calling the function teams_player_played_for() for getting the performace stats.
    details['Matches played for each team'] = teams_player_played_for(Id)
    
    return details


#######################################-- PLAYER Performance --###########################################

def GetPerformanceStat(id , N = 5):

    # Getting a DataFrame subset with stats of a player in last N matches.
    df = Player_data[Player_data.player_id == id].sort_values(by=['match_id'])[['player_id','match_id','team_id','player_name','player_total_points','player_raid_points_total','player_tackle_points_total','player_raids_total','player_raids_successful','player_raids_unsuccessful','player_raids_empty','player_tackles_total','player_tackles_successful','player_tackles_unsuccessful']][-N:]
    
    # Getting names of oposition teams in last N matches.
    op_teams = []
    for i in df[['match_id','team_id']].values:
        op_teams.append( str(len(op_teams)+1) +" " + Team_data[(Team_data.match_id == i[0]) & (Team_data.id != i[1])].short_name.values[0])

    df['Oposition'] = op_teams
    df.set_index(['Oposition'] ,  inplace = True)
    df.sort_values("match_id", axis = 0, ascending = True, inplace = True)
    
    # Getting name of the player.
    name = Player_data[Player_data.player_id == id].player_name.values[0] 

    return df , name
    

######################################-- Teams Stats --####################################################
def GetTeams_Stats():

    teams_win = pd.DataFrame()
    for i in team_ids:
        teams_win = pd.concat([teams_win , Team_data[Team_data.id == i]])
    # teams_win.rename(columns={"Unnamed: 0": "win"} , inplace=True)
    
    teams_win['win'] = teams_win.index 
    
    teams_win.reset_index(drop = True , inplace = True)
    
    new_data = teams_win[['win', 'id', 'match_id', 'name', 'score', 'short_name']]

    # Looping through each matchid.
    for i in new_data.match_id.unique():

        # ind variable will be assigned with 2 index values of rows that have same match_id.
        ind = new_data[new_data.match_id == i].index

        # compairing the scores at both locs and assigned 1 to the team with greater score.
        # and 0 to team less score. if the scores are same we have assigned both teams to 2.
        if new_data.loc[ind[0] ,'score'] < new_data.loc[ind[1] ,'score'] :
            new_data.loc[ind[1] ,'win'] = 1
            new_data.loc[ind[0] ,'win'] = 0

        elif new_data.loc[ind[0] ,'score'] == new_data.loc[ind[1] ,'score'] :
            new_data.loc[ind[1] ,'win'] = 2
            new_data.loc[ind[0] ,'win'] = 2

        else:
            new_data.loc[ind[1] ,'win'] = 0
            new_data.loc[ind[0] ,'win'] = 1

    # Creating a dataframe that contains counts of wins, loos, and ties of each team.
    Teams_stats = pd.DataFrame(new_data[new_data.id == 1].win.value_counts()).T
    names=[teams_catalog[teams_catalog.team_id == 1].short_name.values[0] ]
    ids = [1]
    total = [new_data[new_data.id == 1].shape[0]]
    for id in team_ids:
        Teams_stats = pd.concat([Teams_stats , pd.DataFrame(new_data[new_data.id == id].win.value_counts()).T])
        ids.append(id)
        total.append( new_data[new_data.id == id].shape[0] )
        names.append( teams_catalog[teams_catalog.team_id == id].short_name.values[0])
    Teams_stats['total'] = total
    Teams_stats['name'] = names
    Teams_stats = Teams_stats.set_index([ids])
    Teams_stats.drop_duplicates( inplace= True)
    return Teams_stats , new_data
    # Teams_stats.rename(columns={0: "Loos", 1: "Win", 2:"Tie"} , inplace = True)
    # st.write(Teams_stats)
    # st.bar_chart(Teams_stats.set_index(["name"]))


def plotTeamsStats(df):
    # Plotting bar plots using above dataFrame.
    x = np.arange(len(df.name))  # the label locations
    width = 0.25  # the width of the bars

    fig = plt.figure(figsize=(12,5))
    ax = fig.subplots()
    rects1 = ax.bar(x - width, df['total'], width, label='Total')
    rects2 = ax.bar(x - width/2, df[1],  width,color = 'g' , label='Win')
    rects3 = ax.bar(x + width/2, df[0], width,color = 'r', label='Loos')
    rects4 = ax.bar(x + width, df[2], width,color = 'orange', label='Tie')

    ax.set_ylabel('Counts')
    ax.set_title('Counts by teams and results in all 7 seasons')
    ax.set_xticks(x)
    ax.set_xticklabels(df.name)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    return fig

#############################################== Teams Performance ==###################################################


def GetTeamPerformanceStat(id , N = 5):
    df = Team_data[Team_data.id == id].sort_values(by= ['match_id'])[[ 'id', 'match_id', 'name', 'score', 'short_name','stats.all_outs',  'stats.points.all_out','stats.points.extras','stats.points.raid_points.raid_bonus', 'stats.points.raid_points.total','stats.points.raid_points.touch', 'stats.points.tackle_points.capture','stats.points.tackle_points.capture_bonus','stats.points.tackle_points.total', 'stats.points.total','stats.raids.empty', 'stats.raids.successful','stats.raids.super_raids', 'stats.raids.total','stats.raids.unsuccessful', 'stats.tackles.successful','stats.tackles.super_tackles', 'stats.tackles.total','stats.tackles.unsuccessful']][-N:]
    op_teams = []
    op_res = []
    
    wins= new_data[new_data.id == id].sort_values(by=['match_id'])[-N:].win.value_counts()
    name = teams_catalog[teams_catalog['team_id'] == id].team_name.values[0]
    for i in df[['match_id','id']].values:
        op_teams.append(str(len(op_teams)+1) +" " +new_data[(new_data.match_id == i[0]) & (new_data.id != i[1])].short_name.values[0])
        op_res.append(new_data[(new_data.match_id == i[0]) & (new_data.id != i[1])][['short_name','win']].values[0])

    df['Oposition'] = op_teams
    df.set_index(['Oposition'] ,  inplace = True)
    df.sort_values("match_id", axis = 0, ascending = True, inplace = True)
    
    # Total points stats
    fig1 = plt.figure(1)
    ax = fig1.add_axes([0,0,1,1])
    ax.bar(df.match_id.values + 0.0, df['stats.points.total'], color = 'b', width = 1)
    ax.bar(df.match_id.values + 0.50, df['stats.points.raid_points.total'], color = 'orange', width = 0.60)
    ax.bar(df.match_id.values + 0.50, df['stats.points.tackle_points.total'] , bottom= df['stats.points.raid_points.total'] , color = 'g', width = 0.60)
    plt.xticks( df.match_id.values, labels=op_teams )
    plt.legend(labels=['Total','Raids', 'tackles'])
    plt.title('Total points stats of '+ name +' in last'+ str(N) +' matches.')
    plt.xlabel('Opponent teams')
    plt.ylabel('Points')
    
    # Pie chart
    fig4 = plt.figure( 4,figsize=(12,5),facecolor='#F2F3F4', linewidth=5,edgecolor='#04253a')
    axes = fig4.subplots(1,N)

    label = ['stats.points.raid_points.total','stats.points.tackle_points.total']
    for i, ax in enumerate(axes.flatten()):
        Raids = df[['stats.points.raid_points.total','stats.points.tackle_points.total']][i:i+1].values[0]
        ax.pie(Raids,colors=['orange','g'], autopct='%1.2f%%' , radius= 1.25)
        
        ax.set_title( op_res[i][0] + ' ' +str(op_res[i][1]))
    plt.legend(labels = label)
    
    
    # Raids Stats
    fig2 = plt.figure(2)
    ax = fig2.add_axes([0,0,1,1])
    ax.bar(df.match_id.values + 0.0, df['stats.raids.total'], color = 'b', width = 1)
    ax.bar(df.match_id.values + 0.50, df['stats.raids.successful'], color = 'orange', width = 0.60)
    ax.bar(df.match_id.values + 0.50, df['stats.raids.unsuccessful'] , bottom= df['stats.raids.successful'] , color = 'r', width = 0.60)
    plt.xticks( df.match_id.values, labels=op_teams)
    plt.legend(labels=['Total_raids','Successful_raids', 'Unsuccessful_raids'])
    plt.title('Raids stats of '+ name +' in last'+ str(N) +' matches.')
    plt.xlabel('Opponent teams')
    plt.ylabel('Counts')
    # Pie chart
    fig5 = plt.figure(figsize=(12,5),facecolor='#F2F3F4', linewidth=5,edgecolor='#04253a')
    axes = fig5.subplots(1,N)
    
    label = ['stats.raids.empty','stats.raids.successful','stats.raids.unsuccessful']
    for i, ax in enumerate(axes.flatten()):
        Raids = df[['stats.raids.empty','stats.raids.successful','stats.raids.unsuccessful']][i:i+1].values[0]
        ax.pie(Raids, colors=['b','orange','r'],autopct='%1.2f%%' , radius= 1.25)
        
        ax.set_title(op_res[i][0] + ' ' +str(op_res[i][1]))
    plt.legend(labels = label)
    


    # Tackel stats
    fig3 = plt.figure(3)
    ax = fig3.add_axes([0,0,1,1])
    ax.bar(df.match_id.values + 0.0, df['stats.tackles.total'], color = 'b', width = 1)
    ax.bar(df.match_id.values + 0.50, df['stats.tackles.successful'], color = 'orange', width = 0.60)
    ax.bar(df.match_id.values + 0.50, df['stats.tackles.unsuccessful'] , bottom= df['stats.tackles.successful'] , color = 'r', width = 0.60)
    plt.xticks( df.match_id.values, labels=op_teams)
    plt.legend(labels=['Total_tackles','Successful_tackles', 'Unsuccessful_tackles'])
    plt.title('Tackle stats of '+ name +' in last'+ str(N) +' matches.')
    plt.xlabel('Opponent teams')
    plt.ylabel('Counts')
    # Pie chart
    fig6 = plt.figure(figsize=(12,5),facecolor='#F2F3F4', linewidth=5,edgecolor='#04253a')
    axes = fig6.subplots(1,N)
     
    label = ['stats.tackles.successful','stats.tackles.unsuccessful']
    for i, ax in enumerate(axes.flatten()):
        Raids = df[['stats.tackles.successful','stats.tackles.unsuccessful']][i:i+1].values[0]
        ax.pie(Raids, colors=['orange','r'], autopct='%1.2f%%'  , radius= 1.25)
        
        ax.set_title(op_res[i][0] + ' ' +str(op_res[i][1]))
    plt.legend(labels = label)
    

    return  wins, df , name , [fig1, fig2 , fig3 , fig4, fig5 , fig6]

#############################################== Compair Teams ==###################################################

def CompareTwoTeams( id1 , id2 , N = 5 ):

    # Getting all thoes matche were these teams played against each other.
    mids = pd.merge(new_data[new_data.id == id1 ],new_data[new_data.id == id2 ] , on = 'match_id').sort_values(by =['match_id']).match_id[-N:].values

    cols = [ 'id', 'match_id', 'name', 'score', 'short_name','stats.all_outs',  'stats.points.all_out','stats.points.extras','stats.points.raid_points.raid_bonus', 'stats.points.raid_points.total','stats.points.raid_points.touch', 'stats.points.tackle_points.capture','stats.points.tackle_points.capture_bonus','stats.points.tackle_points.total', 'stats.points.total','stats.raids.empty', 'stats.raids.successful','stats.raids.super_raids', 'stats.raids.total','stats.raids.unsuccessful', 'stats.tackles.successful','stats.tackles.super_tackles', 'stats.tackles.total','stats.tackles.unsuccessful']
    df1 = pd.DataFrame(columns = cols)
    df2 = pd.DataFrame(columns = cols)
    for i in mids:
        df1 = pd.concat([ df1 , pd.DataFrame(Team_data[(Team_data.id == id1) & ( Team_data.match_id == i)][cols])])
        df2 = pd.concat([ df2 , pd.DataFrame(Team_data[(Team_data.id == id2) & ( Team_data.match_id == i)][cols])])

    # getting names og both the teams.
    t1 = teams_catalog[teams_catalog.team_id == id1].team_name.values[0]
    t2 = teams_catalog[teams_catalog.team_id == id2].team_name.values[0]

    # Plotting bar plots using above dataFrame.
    x = np.arange(len(mids))  # the label locations
    width = 0.13  # the width of the bars

    fig = plt.figure(figsize=(20,5), linewidth=5,edgecolor='#04253a')
    ax = fig.subplots( )
    rects1 = ax.bar(x - width*2.5, df1['stats.points.total'],  width,color = 'b', label='T1 :'+ t1 +' Total')
    rects2 = ax.bar(x - width*1.5, df1['stats.points.raid_points.total'],  width,color = 'orange',  label=t1 +' Raids')
    rects3 = ax.bar(x - width/2, df1['stats.points.tackle_points.total'] ,  width, color = 'r', label=t1 +' Tackles')
    rects4 = ax.bar(x + width/2, df2['stats.points.total'],  width,color = 'g', label='T2 :'+t2 +' Total')
    rects5 = ax.bar(x + width*1.5, df2['stats.points.raid_points.total'],  width,color = 'orange',  label=t2 +' Raids')
    rects6 = ax.bar(x + width*2.5, df2['stats.points.tackle_points.total'] ,  width,color = 'r', label=t2 +' Tackles')

    ax.set_ylabel('Points')
    ax.set_xlabel('Match ids')
    ax.set_title('Points by Both teams in the matches.')
    ax.set_xticks(x)
    ax.set_xticklabels(mids)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    fig.tight_layout()

    # pie charts
    fig2 = plt.figure(figsize=(20,5),facecolor='#F2F3F4', linewidth=5,edgecolor='#04253a')
    axes = fig2.subplots(2,len(mids))

    label = ['stats.raids.empty','stats.raids.successful','stats.raids.unsuccessful']
    for i, ax in enumerate(axes[0].flatten()):
        Raids = df1[['stats.raids.empty','stats.raids.successful','stats.raids.unsuccessful']][i:i+1].values[0]
        ax.pie(Raids, colors=['b','orange','r'],autopct='%1.2f%%' , radius= 1.25)
        
        # ax.set_title(op_res[i][0] + ' ' +str(op_res[i][1]))
        ax.set_title(str(id1)+' : Match_id :'+str(mids[i]))


    for i, ax in enumerate(axes[1].flatten()):
        Raids = df2[['stats.raids.empty','stats.raids.successful','stats.raids.unsuccessful']][i:i+1].values[0]
        ax.pie(Raids, colors=['b','orange','r'],autopct='%1.2f%%' , radius= 1.25)
        
        # ax.set_title(op_res[i][0] + ' ' +str(op_res[i][1]))
        ax.set_title(str(id2)+' : Match_id :'+str(mids[i]))

    plt.legend(labels = label)

    # pie charts
    fig3 = plt.figure(figsize=(20,5), linewidth=5,edgecolor='#04253a')
    axes = fig3.subplots(2,len(mids))

    label = ['stats.tackles.successful','stats.tackles.unsuccessful']
    for i, ax in enumerate(axes[0].flatten()):
        Tackles = df1[['stats.tackles.successful','stats.tackles.unsuccessful']][i:i+1].values[0]
        ax.pie(Tackles, colors=['orange','r'],autopct='%1.2f%%' , radius= 1.25)
        
        # ax.set_title(op_res[i][0] + ' ' +str(op_res[i][1]))
        ax.set_title(str(id1)+' : Match_id :'+str(mids[i]))


    for i, ax in enumerate(axes[1].flatten()):
        Tackles = df2[['stats.tackles.successful','stats.tackles.unsuccessful']][i:i+1].values[0]
        ax.pie(Tackles, colors=['orange','r'],autopct='%1.2f%%' , radius= 1.25)
        
        # ax.set_title(op_res[i][0] + ' ' +str(op_res[i][1]))
        ax.set_title(str(id2)+' : Match_id :'+str(mids[i]))

    plt.legend(labels = label)

    return fig , fig2 , fig3


if __name__ == "__main__":
    

    st.title('Exploratory Data Analysis Pro Kabbadi all 7 seasons data')

    DATE_COLUMN = 'date'
    MATCH_URL = ('https://raw.githubusercontent.com/ranganadhkodali/Pro-Kabadi-season-1-7-Stats/master/DS_match.csv')
    PLAYER_URL = ('https://raw.githubusercontent.com/ranganadhkodali/Pro-Kabadi-season-1-7-Stats/master/DS_players.csv')
    TEAM_URL = ('https://raw.githubusercontent.com/ranganadhkodali/Pro-Kabadi-season-1-7-Stats/master/DS_team.csv')

    data_load_state = st.text('Loading data...')
    Match_data = load_data(MATCH_URL)
    Player_data = load_data(PLAYER_URL)
    Team_data = load_data(TEAM_URL)
    data_load_state.text("Done! (using st.cache)")

    #### Creating Teams Catalog......

    team_ids = sorted(Team_data.id.value_counts()[:12].index)

    teams_catalog = pd.DataFrame()
    Id =  []
    names = []
    sn = []
    for id in team_ids:
        Id.append(id)
        names.append(Team_data[Team_data.id == id].name.unique()[0])
        sn.append(Team_data[Team_data.id == id].short_name.unique()[0])
    teams_catalog['team_id'] = Id
    teams_catalog['team_name'] = names
    teams_catalog['short_name'] = sn
    teams_catalog.sort_values(by=['team_id'] , inplace = True)

    #### Creating Players Catalog......

    player_ids = []
    for i in team_ids:
        player_ids.extend( Player_data[Player_data.team_id == i].player_id.value_counts().index)
    player_ids = set(player_ids)

    players_catalog = pd.DataFrame()
    Id =  []
    names = []
    for id in player_ids:
        Id.append(id)
        names.append(Player_data[Player_data.player_id == id].player_name.unique()[0])
    players_catalog['player_id'] = Id
    players_catalog['player_name'] = names
    players_catalog.sort_values(by=['player_id'] , inplace = True)

    teams = Team_data[['id','name','short_name']].drop_duplicates(['id'], keep='last' )

    new_data = GetTeams_Stats()[1]



    #####################################===SIDE BAR===##################################################

    st.sidebar.title('Navigation')
    nav = st.sidebar.radio("Project Content",('Objective','Players','Teams','Code'))

    if nav == 'Objective':
        image = Image.open('./images/kabbadi.jpg')
        st.image(image, caption='Kabbadi Poster/Banner', use_column_width=True)
        st.markdown('''
        ## Players :

        * Where this results can be help full : 
        * At auction were teams bid for players.
        * To get insights of opponent teams.

        * Record Holders : Created Different functions to get. functions takes N as input to return that no. of players info like (Id, Name, Score).
        * Top raiders : who scored most in raids in a match. 
        * Top defenders : who scored most in tackles in a match.
        * Top AllRounder : who scored most in both in a match.

        * Players whole info. : A function that takes the player id as input and return the details like.
        * Name 
        * Total matches played (shows experience) 
        * Stats of performance in all the teams that player played for(DataFrame).
            * Team name ( from teams data file)
            * no. of matches played ( from player data file)
            * performance include successful and unsuccessful (raids and tackles)
            * Visualization of Players performance in recent N matches.

        ## Teams :

        * Visualization of Teams winning to loosing ratio in all pro kabbadi seasons.

        * Building a model for predicting wheather the combination of players will win or loos the match.

        * Visualization of Teams performance in recent N matches.

        * Visualization of comparing two Teams performance against each other in recent N matches.
        
        ''')
        

        st.header('The Data used for this Project')
        
        if st.checkbox('Show Matches data'):
            st.header('Matches data')
            st.dataframe(Match_data)
        if st.checkbox('Show Players data'):
            st.header('Players data')
            st.dataframe(Player_data)
        if st.checkbox('Show Teams data'):
            st.header('Teams data')
            st.dataframe(Team_data)
        st.info('''NOTE : Data is been Cleaned & Processed well for this implementation.
        \n Got the data from : https://github.com/ranganadhkodali/Pro-Kabadi-season-1-7-Stats''')


    ################################################=== Players Section ==########################################################


    elif nav == 'Players':
        st.title('Players')

        if st.checkbox('Show Players Catalog'):
            st.header('Players data')
            st.dataframe(players_catalog)

        st.header('Filter for Top Players')
        category = st.selectbox('Select the category',['Allrounders', 'Raiders','Defenders'])

        topn = st.slider( 'Select number of players' , 5 , 25 , 10)

        st.subheader(('Top '+str(topn)+' '+category + ' Scores'))

        if st.checkbox('Show Unique Players from the list'):
            res , lab = GetTopPlayers(category , topn)
        else:
            res , lab = GetTopScores(category , topn)
        st.write(lab)
        st.dataframe(res)

        
        st.write('_______________________________________________________________________________________________')
        st.header("Search for a Player's Details. " )
        PL_ID = st.text_input("Player's ID you want to get Details of goes here...", 81)

        information = Get_player_details(int(PL_ID))
        pt = './images/'+str(PL_ID)+'.jpg'
        if path.exists(pt):
            image = Image.open(pt)
            st.image(image, caption=information['Name'] , use_column_width=True)

        st.success( 'NAME : '+ information['Name'])

        st.success( 'Total No. of matches played : ' + str(information['Total No. of matches played']))

        st.subheader( 'Stats of Matches played for each team ')
        st.table(information['Matches played for each team'])

        st.write('_______________________________________________________________________________________________')
        st.header('Select number of recent matches.')
        rn = st.slider("", 1 , 15 , 5)
        df , name = GetPerformanceStat(int(PL_ID), rn)

        st.header(name + "'s Performance Stats")
        st.subheader('Select the options you want to see stats for')
        chioce = st.radio('', ('Overall','Raids','Tackles') )
        
        if chioce == 'Overall':
            # Total points stats
            st.subheader('Total points stats in recent '+str(rn)+' Matches')
            if st.checkbox('Show Data'):
                st.write(df[['player_total_points','player_raid_points_total','player_tackle_points_total']].T )
            if st.checkbox('Show Bar Chart'):
                st.bar_chart(df[['player_total_points','player_raid_points_total','player_tackle_points_total']] )
            if st.checkbox('Show Area Chart'):
                st.area_chart(df[['player_total_points','player_raid_points_total','player_tackle_points_total']])

        elif chioce == 'Raids':
            # Raids Stats
            st.subheader('Total Raids Statsin recent '+str(rn)+' Matches')
            if st.checkbox('Show Data'):
                st.write(df[['player_raids_total','player_raids_successful','player_raids_empty','player_raids_unsuccessful']].T )
            if st.checkbox('Show Bar Chart'):
                st.bar_chart(df[['player_raids_total','player_raids_successful','player_raids_empty','player_raids_unsuccessful']] )
            if st.checkbox('Show Area Chart'):
                st.area_chart(df[['player_raids_total','player_raids_successful','player_raids_empty','player_raids_unsuccessful']] )

        elif chioce == 'Tackles':
            # Tackel stats
            st.subheader('Total Raids Statsin recent '+str(rn)+' Matches')
            if st.checkbox('Show Data'):
                st.write(df[['player_tackles_total','player_tackles_successful','player_tackles_unsuccessful']].T )
            if st.checkbox('Show Bar Chart'):
                st.bar_chart(df[['player_tackles_total','player_tackles_successful','player_tackles_unsuccessful']] )
            if st.checkbox('Show Area Chart'):
                st.area_chart(df[['player_tackles_total','player_tackles_successful','player_tackles_unsuccessful']] )
            

    ################################################=== Teams Section ==########################################################


    elif nav == 'Teams':
        st.title('Teams')
        if st.checkbox('Show Teams Catalog'):
            st.header('Teams data')
            st.table(teams_catalog)

        st.header('All Teams Stats')
        df = GetTeams_Stats()[0]
        tc = st.radio('',('Show Data' , 'Graph'))
        if tc == 'Show Data':
            st.subheader('Data of all Teams match counts and results.')
            st.table(df.rename(columns={0: "Loos", 1: "Win", 2:"Tie"} ))
        elif tc == 'Graph':
            st.subheader('Graphical representation of Data')
            st.pyplot(plotTeamsStats(df))
            st.info('''
            ### **Conclusion**

            * Here we can see that teams (  PAT , MUM , and GFG) won more matches than loosing.
            * Teams( BEN ,JAI , KOL , HYD , UPY , HS) has difference in no. of winning and loosing  matches.
            * Teams( PUN , DEL , TT ) loss more matches that winning.''')
        
        st.write('________________________________________________________________________________________________________')
        st.header("Search for a Team's Performance. " )
        # T_ID = st.text_input("Team's ID you want to get Details of goes here...", 1)
        T_ID = st.selectbox("Team's ID you want to get Details of goes here...",teams_catalog.team_name.values)
        T_ID = teams_catalog[ teams_catalog.team_name == T_ID ].team_id.values[0]
        st.subheader('Select number of recent matches.')
        rm = st.slider("", 1 , 10 , 5)

        w , df , name , figs = GetTeamPerformanceStat( T_ID , rm)

        T_choice = st.radio('', ('Overall','Raids','Tackles') )

        
        
        if T_choice == 'Overall':
            # Total points stats
            st.subheader('Total points stats in Last '+str(rm)+' Matches')
            if st.checkbox('Show Data'):
                st.write(df[['stats.points.total','stats.points.raid_points.total','stats.points.tackle_points.total']].T )
            if st.checkbox('Show Bar Chart'):
                st.pyplot(figs[0])
            if st.checkbox('Show Pie Chart'):
                st.pyplot(figs[3])
                st.info(' * These Pie charts tells us playing statergy of the team like more into rading or defending. ')

        elif T_choice == 'Raids':
            # Raids Stats
            st.subheader('Total Raids Statsin Last '+str(rm)+' Matches')
            if st.checkbox('Show Data'):
                st.write(df[['stats.raids.total','stats.raids.empty','stats.raids.successful','stats.raids.unsuccessful']].T )
            if st.checkbox('Show Bar Chart'):
                st.pyplot(figs[1])
            if st.checkbox('Show Pie Chart'):
                st.pyplot(figs[4])
                st.info(" * These Pie charts tells us how the team's raiders performed." )


        elif T_choice == 'Tackles':
            # Tackel stats
            st.subheader('Total Raids Statsin Last '+str(rm)+' Matches')
            if st.checkbox('Show Data'):
                st.write(df[['stats.tackles.total','stats.tackles.successful','stats.tackles.unsuccessful']].T )
            if st.checkbox('Show Bar Chart'):
                st.pyplot(figs[2])
            if st.checkbox('Show Pie Chart'):
                st.pyplot(figs[5])
                st.info(" * These Pie charts tells us how the team's defenders performed." )

    
        st.write('________________________________________________________________________________________________________')
        st.header("Compair the Performance of two Teams. " )
        T_ID1 = st.selectbox('Select the Team 1',teams_catalog.team_name.values)
        T_ID1 = teams_catalog[ teams_catalog.team_name == T_ID1 ].team_id.values[0]
        T_ID2 = st.selectbox('Select the Team 2',teams_catalog[teams_catalog.team_id != T_ID1].team_name.values)
        T_ID2 = teams_catalog[ teams_catalog.team_name == T_ID2 ].team_id.values[0]
        # T_ID2 = st.text_input("Team's ID you want to get Details of goes here...", 5)
        st.subheader('Select number of recent matches.')
        n = st.slider('\/', 1 , 10 , 5)

        fig1 , fig2 , fig3 = CompareTwoTeams( T_ID1 , T_ID2 , n )

        st.pyplot(fig1)
        st.info(''' * In this bar chart we are compairing the scores.
        * The Blue and Green bars tells us scores in that match.
        * And the Yellow and the Red bars represents the total Raid and Teckel points scored in that match.
        ''' )

        st.pyplot(fig2)
        st.info(''' * In this pie chart we are compairing the Raid stats.
        * The Blue arc represents Empty raids in that match.
        * And the Yellow arc represents Sucessfull raids in that match.
        * And the Red arc represents Unsucessfull raids in that match.
        ''' )
        st.pyplot(fig3)
        
        st.info(''' * In this pie chart we are compairing the Tackels stats.
        * The Yellow arc represents Sucessfull Tackels in that match.
        * And the Red arc represents Unsucessfull Tackels in that match.
        ''' )
    ################################################=== Code Section ==########################################################
    elif nav == 'Code':
        file = open("PKcode.txt","r") 
        code = file.read()
        st.title('Code written for this project...:sunglasses:')
        with st.echo():
                code

