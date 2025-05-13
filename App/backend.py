
import pymysql
pymysql.install_as_MySQLdb()
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from itertools import product
import base64
import os
import google.generativeai as genai

conn_string = 'mysql://{user}:{password}@{host}:{port}/{db}?charset={encoding}'.format(
    user='Team_A',
    password='NkC121jpeTE=',
    host = 'jsedocc7.scrc.nyu.edu',
    port = 3306,
    encoding = 'utf8',
    db = 'Team_A'
)
engine = create_engine(conn_string)

TEAM_CODE_TO_NAME = dict([
    ('ATL', 'Atlanta Hawks'), ('BOS', 'Boston Celtics'), ('BKN', 'Brooklyn Nets'),
    ('CHA', 'Charlotte Hornets'), ('CHI', 'Chicago Bulls'), ('CLE', 'Cleveland Cavaliers'),
    ('DAL', 'Dallas Mavericks'), ('DEN', 'Denver Nuggets'), ('DET', 'Detroit Pistons'),
    ('GSW', 'Golden State Warriors'), ('HOU', 'Houston Rockets'), ('IND', 'Indiana Pacers'),
    ('LAC', 'Los Angeles Clippers'), ('LAL', 'Los Angeles Lakers'), ('MEM', 'Memphis Grizzlies'),
    ('MIA', 'Miami Heat'), ('MIL', 'Milwaukee Bucks'), ('MIN', 'Minnesota Timberwolves'),
    ('NOP', 'New Orleans Pelicans'), ('NYK', 'New York Knicks'), ('OKC', 'Oklahoma City Thunder'),
    ('ORL', 'Orlando Magic'), ('PHI', 'Philadelphia 76ers'), ('PHX', 'Phoenix Suns'),
    ('POR', 'Portland Trail Blazers'), ('SAC', 'Sacramento Kings'), ('SAS', 'San Antonio Spurs'),
    ('TOR', 'Toronto Raptors'), ('UTA', 'Utah Jazz'), ('WAS', 'Washington Wizards')
])

playerSeasonStats = pd.read_sql("SELECT * FROM playerSeasonStats", con=engine)
teamSeasonData = pd.read_sql("SELECT * FROM teamSeasonData", con=engine)
teamSchedule = pd.read_sql("SELECT * FROM teamSchedule", con=engine)
all_players_gamelogs = pd.read_sql("SELECT * FROM all_players_gamelogs", con=engine)

def to_base64_img():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"

"""## Player Performance"""
def getPlayerStats(playerName):
    playerStats = all_players_gamelogs[(all_players_gamelogs['PLAYER_NAME'] == playerName) & (all_players_gamelogs['MIN'] > 0)]
    return playerStats

def graphPlayerScatterPlot(playerName, stat):
    """
    Graphs a scatter plot of a player's performance for a specific stat
    """
    playerStats = getPlayerStats(playerName)
    if playerStats.empty:
      print(f"No data found for player: {playerName}")
      return

    plt.figure(figsize=(10, 6))
    playerStats['GAME_DATE_NUM'] = (playerStats['GAME_DATE'] - playerStats['GAME_DATE'].min()).dt.days

    plt.scatter(
        playerStats['GAME_DATE_NUM'],
        playerStats[stat],
        c=playerStats['WL'].map({'W': 'green', 'L': 'red'}),
        zorder=3,
        label='Game Result'
    )

    sns.regplot(
        x='GAME_DATE_NUM',
        y=stat,
        data=playerStats,
        line_kws={'color': 'blue', 'lw': 2},
        ci=None,
        label="Line of Best Fit"
    )
    plt.title(f"{playerName}'s {stat} This Season")
    plt.xlabel('Game Date')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(stat)
    plt.tight_layout()
    return to_base64_img()

def graphPlayerDensityPlot(playerName, stat):
    """
    Graphs a density plot of a player's performance for a specific stat
    """
    playerStats = getPlayerStats(playerName)
    if playerStats.empty:
        print(f"No data found for player: {playerName}")
        return

    scatterplot = playerStats.plot(kind='scatter', x='GAME_DATE', y=stat, figsize=(10, 6))
    plt.scatter(
        playerStats['GAME_DATE'],
        playerStats[stat],
        c=playerStats['WL'].map({'W': 'green', 'L': 'red'}),
        zorder=3
    )

    sns.kdeplot(x=playerStats['GAME_DATE'], y=playerStats[stat], cmap=plt.cm.Blues, ax = scatterplot)
    plt.title(f"Density Plot of {playerName}'s {stat}")
    plt.xlabel('Game Date')
    plt.ylabel(stat)
    plt.tight_layout()
    return to_base64_img()

# def graphTimeSeriesPlot(playerName, stat):
#     """
#     Graphs a time series plot of a player's performance for a specific stat
#     """
#     playerStats = getPlayerStats(playerName)
#     if playerStats.empty:
#         print(f"No data found for player: {playerName}")
#         return
    
#     playerStats = playerStats.copy()
#     playerStats['RollingMedian'] = playerStats[stat].rolling(window=4).median()

#     plt.figure(figsize=(10, 6))
#     plt.plot(playerStats['GAME_DATE'], playerStats[stat], label=stat)
#     plt.plot(playerStats['GAME_DATE'], playerStats['RollingMedian'], label='Rolling Median', linestyle='--')

#     plt.title(f"{playerName}'s {stat} Over Time")
#     plt.xlabel("Game Date")
#     plt.ylabel(stat)
#     plt.legend()
#     plt.tight_layout()
#     return to_base64_img()
def graphTimeSeriesPlot(playerName, stat):
    """
    Graphs a time series plot of a player's performance for a specific stat using Plotly
    """
    playerStats = getPlayerStats(playerName)
    if playerStats.empty:
        print(f"No data found for player: {playerName}")
        return ""

    playerStats = playerStats.copy()
    playerStats['RollingMedian'] = playerStats[stat].rolling(window=4).median()

    fig = go.Figure()

    # Original stat line
    fig.add_trace(go.Scatter(
        x=playerStats['GAME_DATE'],
        y=playerStats[stat],
        mode='lines+markers',
        name=stat
    ))

    # Rolling median line
    fig.add_trace(go.Scatter(
        x=playerStats['GAME_DATE'],
        y=playerStats['RollingMedian'],
        mode='lines',
        name='Rolling Median',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title=f"{playerName}'s {stat} Over Time",
        xaxis_title='Game Date',
        yaxis_title=stat,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_white',
        height=500
    )

    return fig.to_html(full_html=False)

def graphTimeSeriesPlotAllStats(playerName):
    """
    Returns a base64-encoded PNG of the player's PTS/REB/AST performance using Matplotlib.
    """
    playerStats = getPlayerStats(playerName)
    if playerStats.empty:
        print(f"No data found for player: {playerName}")
        return None

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    for stat in ['PTS', 'REB', 'AST']:
        plt.plot(playerStats['GAME_DATE'], playerStats[stat], label=stat)

    plt.xlabel("Game Date")
    plt.ylabel("Stat Value")
    plt.title(f"{playerName}'s Performance Over Time")
    plt.legend()
    plt.tight_layout()
    return to_base64_img()

"""## Players Ranked against each other"""

def rank_players_by_stat(stat, ascending=False):
    """
    Ranks players by a given stat
    """
    filtered_df = all_players_gamelogs[all_players_gamelogs['MIN'] > 0].copy()
    player_avg = filtered_df.groupby('PLAYER_NAME')[stat].mean().reset_index()
    player_avg = player_avg.sort_values(by=stat, ascending=ascending).reset_index(drop=True)
    player_avg['Rank'] = player_avg.index + 1
    return player_avg


def composite_ranking(stats=['PTS', 'REB', 'AST']):
    """
    Creates a composite ranking for players based on multiple stats.
    """
    filtered_df = all_players_gamelogs[all_players_gamelogs['MIN'] > 0].copy()
    player_stats = filtered_df.groupby('PLAYER_NAME')[stats].mean().reset_index()

    for stat in stats:
         player_stats[stat + '_z'] = (player_stats[stat] - player_stats[stat].mean()) / player_stats[stat].std()
    player_stats['Composite'] = player_stats[[stat + '_z' for stat in stats]].sum(axis=1)

    player_stats = player_stats.sort_values(by='Composite', ascending=False).reset_index(drop=True)
    player_stats['Rank'] = player_stats.index + 1

    columns_to_show = ['PLAYER_NAME', 'Composite', 'Rank'] + stats
    return player_stats[columns_to_show]

def graph_ranking_bar(df, value_col, title, top_n=10):
    """
    Creates a horizontal bar chart of the top ranked players based on a specified value column.
    """
    top_df = df.head(top_n)

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        x=value_col,
        y='PLAYER_NAME',
        data=top_df,
        palette='viridis'
    )

    for i, v in enumerate(top_df[value_col]):
        bar_plot.text(v + 0.01 * v, i, f"{v:.2f}", color='black', va='center')

    plt.title(title, fontsize=14)
    plt.xlabel(value_col, fontsize=12)
    plt.ylabel('Player Name', fontsize=12)
    plt.tight_layout()
    plt.show()

"""## Team Graphs"""
def composite_team_ranking():
    """
    Calculates a composite performance score for each team based on key stats.
    The composite score is computed as:
        z(PTS) + z(REB) + z(AST) + z(STL) + z(BLK) - z(Turnovers)
    """
    df = teamSeasonData.copy()

    required_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'Turnovers']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from teamSeasonData.")

    df['PTS_z'] = (df['PTS'] - df['PTS'].mean()) / df['PTS'].std()
    df['REB_z'] = (df['REB'] - df['REB'].mean()) / df['REB'].std()
    df['AST_z'] = (df['AST'] - df['AST'].mean()) / df['AST'].std()
    df['STL_z'] = (df['STL'] - df['STL'].mean()) / df['STL'].std()
    df['BLK_z'] = (df['BLK'] - df['BLK'].mean()) / df['BLK'].std()
    df['Turnovers_z'] = (df['Turnovers'] - df['Turnovers'].mean()) / df['Turnovers'].std()

    df['Composite'] = (df['PTS_z'] + df['REB_z'] + df['AST_z'] +
                       df['STL_z'] + df['BLK_z'] - df['Turnovers_z'])

    df = df.sort_values(by='Composite', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1

    return df

def graph_team_composite_ranking(top_n=10):
    """
    Graphs a horizontal bar chart of the top teams based on the composite performance score.
    """
    df = composite_team_ranking()
    top_df = df.head(top_n)

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        x='Composite',
        y='Team',
        data=top_df,
        palette='viridis'
    )

    for i, v in enumerate(top_df['Composite']):
        bar_plot.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')

    plt.title("Top Teams by Composite Performance Score", fontsize=14)
    plt.xlabel("Composite Score", fontsize=12)
    plt.ylabel("Team", fontsize=12)
    plt.tight_layout()
    return to_base64_img()

def graph_team_scatter():
    """
    Creates a scatter plot of each team's Points per Game vs. Composite Performance Score.
    Each point is annotated with the team name.
    """
    df = composite_team_ranking()

    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df,
        x='PTS',
        y='Composite',
        s=100,
        color='blue'
    )

    for idx, row in df.iterrows():
        plt.text(row['PTS'], row['Composite'], row['Team'], fontsize=9, ha='right', va='bottom')

    plt.title("Team Composite Score vs. Points per Game", fontsize=14)
    plt.xlabel("Points per Game (PTS)", fontsize=12)
    plt.ylabel("Composite Score", fontsize=12)
    plt.tight_layout()
    return to_base64_img()

def plot_radar_chart(df, team, metrics):
    """
    Plots a radar chart for a single team across the specified metrics.
    """
    team_data = df[df['Team'] == team][metrics].iloc[0]

    normalized = (team_data - df[metrics].min()) / (df[metrics].max() - df[metrics].min())
    values = normalized.values.tolist()
    values += values[:1]

    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], metrics)

    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2","0.4","0.6","0.8"], color="grey", size=7)
    plt.ylim(0,1)

    ax.plot(angles, values, linewidth=2, linestyle='solid', label=team)
    ax.fill(angles, values, alpha=0.4)

    plt.title(f"{team} Performance Radar Chart", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    return to_base64_img()

def plot_parallel_coordinates(df, metrics, class_column='Team'):
    """
    Plots a parallel coordinates chart for multiple teams.
    """
    df_norm = df.copy()
    for col in metrics:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

    top_df = df_norm.sort_values(by=metrics[0], ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(top_df[[class_column] + metrics], class_column, colormap='viridis')
    plt.title("Parallel Coordinates Plot for Top Teams")
    plt.xlabel("Metrics")
    plt.ylabel("Normalized Value")
    plt.tight_layout()
    return to_base64_img()

def plot_team_heatmap(df, team_name, metrics = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'Turnovers']):
    """
    Plots a heatmap of the metrics for a specific team.
    """
    # Filter team data
    team_data = df[df['Team'] == team_name][metrics].iloc[0]

    # Calculate league averages
    league_avg = df[metrics].mean()

    # Calculate difference from league average
    diff = team_data - league_avg

    # Convert to DataFrame for heatmap
    diff_df = pd.DataFrame(diff, columns=[team_name])

    # Plot heatmap
    plt.figure(figsize=(6, len(metrics)*0.5))
    sns.heatmap(diff_df, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title(f"{team_name} vs League Average (Heatmap)")
    plt.tight_layout()
    return to_base64_img()

def process_team_schedule(team_name):
    """
    Processes the teamSchedule DataFrame for a given team.
    """
    games = teamSchedule[(teamSchedule['hometeam'] == team_name) | (teamSchedule['awayteam'] == team_name)].copy()
    games['datetime'] = pd.to_datetime(games['datetime'])

    def parse_score(row):
        try:
            parts = row['score'].split('-')
            away_score = float(parts[0])
            home_score = float(parts[1])
        except Exception as e:
            away_score, home_score = None, None
        if row['hometeam'] == team_name:
            team_score = home_score
            opp_score = away_score
        elif row['awayteam'] == team_name:
            team_score = away_score
            opp_score = home_score
        else:
            team_score, opp_score = None, None
        return pd.Series({'team_score': team_score, 'opp_score': opp_score})

    scores = games.apply(parse_score, axis=1)
    games = pd.concat([games, scores], axis=1)

    games['PointDiff'] = games['team_score'] - games['opp_score']
    games['ComputedResult'] = games['PointDiff'].apply(lambda x: 'W' if x > 0 else ('L' if x < 0 else 'T'))
    games.sort_values(by='datetime', inplace=True)
    games['GameNumber'] = range(1, len(games) + 1)
    return games

def plot_team_game_performance(team_name):
    """
    Plots the game-by-game performance for a given team.
    """
    games = process_team_schedule(team_name)

    plt.figure(figsize=(12, 6))

    plt.plot(games['GameNumber'], games['PointDiff'], marker='o', linestyle='-', label='Point Differential')

    palette = {'W': 'green', 'L': 'red', 'T': 'blue'}
    sns.scatterplot(x='GameNumber', y='PointDiff', hue='ComputedResult', data=games, palette=palette, s=100, legend='brief')

    plt.axhline(0, color='grey', linestyle='--')
    plt.title(f"{team_name} Game-by-Game Performance", fontsize=16)
    plt.xlabel("Game Number", fontsize=14)
    plt.ylabel("Point Differential", fontsize=14)
    plt.legend(title="Result")
    plt.tight_layout()
    return to_base64_img()

"""## Machine Learning for Player Prediction Vs Team"""

def filter_games_vs_opponent(playerGameLog, opponent_team):
    """
    Filters the playerGameLog for games played against a specific opponent
    """
    # Get player's team abbreviation
    playerName = playerGameLog['PLAYER_NAME'].iloc[0]
    player_team = playerSeasonStats[playerSeasonStats['PLAYER_NAME'] == playerName].copy()['TEAM_ABBREVIATION'].iloc[0]

    # Merge game log with schedule on date
    merged = pd.merge(playerGameLog, teamSchedule, left_on='GAME_DATE', right_on='datetime', how='inner')

    # Filter for games where player's team played the opponent
    mask = (
        ((merged['hometeam'] == player_team) & (merged['awayteam'] == opponent_team)) |
        ((merged['awayteam'] == player_team) & (merged['hometeam'] == opponent_team))
    )

    filtered = merged[mask]

    # Drop columns from teamSchedule
    filtered = filtered[playerGameLog.columns]

    return filtered

def playerPredictionWithRegression(player, opponentTeam, stat):
    """
    Trains a model to predict player performance against a team
    """
    # Features and target
    playerGameLog = all_players_gamelogs[all_players_gamelogs['PLAYER_NAME'] == player]
    X = playerGameLog.drop(columns=['GAME_DATE', 'WL', 'PLAYER_NAME', stat])
    y = playerGameLog[stat]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train model with Linear Regression
    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # Train model with SVR
    # model = svm.SVR(kernel="linear")
    # model.fit(X_train, y_train)

    # Train model with Ridge Regression
    model = linear_model.Ridge(alpha=.5)
    model.fit(X_train, y_train)

    # Train model with neural network
    # model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    # model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}\n")

    # Get df of player games vs a team
    filtered_games = filter_games_vs_opponent(playerGameLog, opponentTeam)
    filtered_games = filtered_games if filtered_games.shape[0] > 0 else playerGameLog

    # Use season average for input
    input_features = filtered_games.drop(columns=['GAME_DATE', 'WL', 'PLAYER_NAME', stat]).mean().to_frame().T
    input_scaled = sc.transform(input_features)

    # Make Prediction
    predicted_stat = model.predict(input_scaled)[0]
    if stat != "PLUS_MINUS":
        predicted_stat = predicted_stat if predicted_stat > 0 else 0
    text = f"Predicted {stat} for {player} vs {opponentTeam} : {predicted_stat:.2f}"
    print(text)
    return text

def playerPredictionWithOverUnder(player, opponentTeam, stat, statNum):
    """
    Trains a model to classify whether a player will exceed a certain threshold for a statistic
    against a specific team.

    Parameters:
    - player: Player name
    - opponentTeam: Opponent team name
    - stat: Statistic to predict (e.g., 'PTS', 'AST', 'REB')
    - statNum: Threshold value for the statistic

    Returns:
    - Text result of the prediction
    """
    # Get player game log
    playerGameLog = all_players_gamelogs[all_players_gamelogs['PLAYER_NAME'] == player]

    # Create binary target: 1 if player exceeds statNum, 0 otherwise
    playerGameLog['TARGET'] = (playerGameLog[stat] >= statNum).astype(int)

    # Features and target
    X = playerGameLog.drop(columns=['GAME_DATE', 'WL', 'PLAYER_NAME', stat, 'TARGET'])
    y = playerGameLog['TARGET']

    # Check if there are at least two classes in the target variable
    if len(np.unique(y)) < 2:
        print(f"Not enough data for prediction for {player} and {stat} with threshold {statNum}. Only one class present in the target variable.")
        return f"Insufficient data for prediction: {player} - {stat} - {statNum}"  # Or any appropriate message

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train model with SVM (Best)
    model = svm.SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    # Other model options (commented out):
    # Train model with Random Forest
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

    # Train model with Logistic Regression
    # model = LogisticRegression(random_state=42)
    # model.fit(X_train, y_train)

    # Train model with neural network
    # model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    # model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Get df of player games vs a team
    filtered_games = filter_games_vs_opponent(playerGameLog, opponentTeam)
    filtered_games = filtered_games if filtered_games.shape[0] > 0 else playerGameLog

    # Use season average for input
    input_features = filtered_games.drop(columns=['GAME_DATE', 'WL', 'PLAYER_NAME', stat, 'TARGET']).mean().to_frame().T
    input_scaled = sc.transform(input_features)

    # Make prediction
    prediction_prob = model.predict_proba(input_scaled)[0][1]  # Probability of class 1
    prediction = model.predict(input_scaled)[0]  # Class prediction (0 or 1)

    # Format the result text
    result = "will" if prediction == 1 else "will not"
    text = f"Prediction: {player} {result} exceed {statNum} {stat} against {opponentTeam} (Probability: {prediction_prob:.2f})"
    print(text)
    return text

"""## Machine Learning for Team Prediction Vs Team"""

#------------------------------------------------------------------
# 1. Helper Functions
#------------------------------------------------------------------
def extract_scores(score_str):
    """
    Given a score string formatted as 'HOME-AWAY' (e.g., '116-120'),
    return a tuple (home_score, away_score) as integers.
    If missing, return (None, None).
    """
    if score_str is None or pd.isnull(score_str):
        return None, None
    try:
        home_score, away_score = map(int, score_str.split('-'))
        return home_score, away_score
    except Exception as e:
        print(f"Error parsing score '{score_str}':", e)
        return None, None

def prepare_simplified_matchup_data(team, teamSeasonData, teamSchedule):
    """
    Constructs a matchup DataFrame for the given team using a simplified feature set.
    For each game the team played (as home or away):
      - Determines the opponent.
      - Computes the target point margin (from the perspective of the team).
      - Computes difference features using selected key statistics:
            diff_PTS, diff_FGPercent, diff_AST, diff_REB, diff_Turnovers
      - Adds an indicator 'is_home' (1 if playing at home, 0 otherwise).
      - Returns a DataFrame with the simplified features, target_margin, and opponent.
    """
    # Filter games where the team is involved and there is a valid score.
    games = teamSchedule.loc[
        (((teamSchedule['hometeam'] == team) | (teamSchedule['awayteam'] == team)) &
         (teamSchedule['score'].notnull()))
    ].copy()

    # We'll use these keys from teamSeasonData:
    keys = ['PTS', 'FGPercent', 'AST', 'REB', 'Turnovers']

    rows = []
    for _, game in games.iterrows():
        home_score, away_score = extract_scores(game['score'])
        if home_score is None or away_score is None:
            continue  # Skip if score parsing fails

        if game['hometeam'] == team:
            is_home = 1
            opponent = game['awayteam']
            target_margin = home_score - away_score
        else:
            is_home = 0
            opponent = game['hometeam']
            target_margin = away_score - home_score

        # Get season stats for both teams.
        team_stats = teamSeasonData.loc[teamSeasonData['Team'] == team]
        opp_stats  = teamSeasonData.loc[teamSeasonData['Team'] == opponent]
        if team_stats.empty or opp_stats.empty:
            continue
        team_stats = team_stats.iloc[0]
        opp_stats = opp_stats.iloc[0]

        feature_dict = {}
        for k in keys:
            feature_dict[f"diff_{k}"] = team_stats[k] - opp_stats[k]
        feature_dict['is_home'] = is_home
        feature_dict['target_margin'] = target_margin
        feature_dict['opponent'] = opponent
        rows.append(feature_dict)

    df = pd.DataFrame(rows)
    return df

#------------------------------------------------------------------
# 2. Combined Model: Regression + Classification (Simplified)
#------------------------------------------------------------------

def teamPredictionCombinedSimplified(team, opponentTeam, teamSeasonData, teamSchedule, is_home=True):
    """
    Builds a combined team‑vs‑team model on a simplified feature set.
    Uses:
      - Ridge Regression to predict point margin.
      - Logistic Regression to predict a win (binary: margin>0).
    Returns:
      A tuple (reg_mse, pred_margin, clf_accuracy, win_prob).
    """
    # Prepare the simplified matchup DataFrame.
    matchup_df = prepare_simplified_matchup_data(team, teamSeasonData, teamSchedule)
    # Optionally, focus on the matchup against a specific opponent.
    data = matchup_df.loc[matchup_df['opponent'] == opponentTeam]
    if data.shape[0] < 5:
        data = matchup_df.copy()

    # Define simplified feature set and target.
    simplified_features = ['diff_PTS', 'diff_FGPercent', 'diff_AST', 'diff_REB', 'diff_Turnovers', 'is_home']

    # For regression: predict point margin.
    X_reg = data[simplified_features]
    y_reg = data['target_margin']

    # For classification: create win target (win if margin > 0).
    data = data.copy()
    data.loc[:, 'win'] = (data['target_margin'] > 0).astype(int)
    X_clf = data[simplified_features]
    y_clf = data['win']

    # Split the data (using 80/20 split) for both tasks.
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # Scale the features.
    scaler_reg = StandardScaler()
    X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_scaled  = scaler_reg.transform(X_reg_test)

    scaler_clf = StandardScaler()
    X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
    X_clf_test_scaled  = scaler_clf.transform(X_clf_test)

    # ----- Regression: Use Ridge Regression -----
    reg_model = Ridge(alpha=1.0, random_state=42)
    reg_model.fit(X_reg_train_scaled, y_reg_train)
    y_reg_pred = reg_model.predict(X_reg_test_scaled)
    reg_mse = mean_squared_error(y_reg_test, y_reg_pred)

    # ----- Classification: Use Logistic Regression -----
    clf_model = LogisticRegression(random_state=42, max_iter=1000)
    clf_model.fit(X_clf_train_scaled, y_clf_train)
    y_clf_pred = clf_model.predict(X_clf_test_scaled)
    clf_accuracy = accuracy_score(y_clf_test, y_clf_pred)

    # Now, predict for the given matchup using season data.
    # Get season stats for team and opponent.
    team_stats = teamSeasonData.loc[teamSeasonData['Team'] == team].iloc[0]
    opp_stats  = teamSeasonData.loc[teamSeasonData['Team'] == opponentTeam].iloc[0]
    keys = ['PTS', 'FGPercent', 'AST', 'REB', 'Turnovers']
    input_features = {}
    for k in keys:
        input_features[f"diff_{k}"] = team_stats[k] - opp_stats[k]
    input_features['is_home'] = 1 if is_home else 0
    input_df = pd.DataFrame([input_features])

    # Scale the new input using the same scalers.
    input_df_reg = scaler_reg.transform(input_df[simplified_features])
    input_df_clf = scaler_clf.transform(input_df[simplified_features])

    pred_margin = reg_model.predict(input_df_reg)[0]
    pred_win_class = clf_model.predict(input_df_clf)[0]
    win_prob = clf_model.predict_proba(input_df_clf)[0][1]  # probability for win=1

    print("\n--- Combined Prediction (Simplified Model) ---")
    print(f"Regression Model MSE: {reg_mse:.2f}")
    print(f"Predicted Point Margin: {pred_margin:.2f}")
    print(f"Classification Accuracy: {clf_accuracy:.2f}")
    print(f"Win Probability: {win_prob:.2f}")

    return reg_mse, pred_margin, clf_accuracy, win_prob

#------------------------------------------------------------------
# 3. Interactive Combined Interface
#------------------------------------------------------------------

def interactive_team_vs_team_model_combined_simplified():
    """
    Interactive Combined Team vs Team Prediction Model using Simplified Features.

    This function does the following:
      1. Prompts the user for the team and opponent abbreviations.
      2. Prompts whether the team is playing at home.
      3. Calls the combined model function that uses:
            - Ridge Regression to predict the point margin.
            - Logistic Regression to estimate the win probability.
      4. Prints:
            - The regression model's MSE and predicted point margin.
            - The classification model's accuracy and win probability.

    Usage:
      Simply run this function (uncomment the call if necessary) to begin the interactive session:

          interactive_team_vs_team_model_combined_simplified()
    """
    print("Combined Team vs Team Prediction Model (Simplified Features)")
    team = input("Enter your team abbreviation (e.g., LAL): ").strip().upper()
    opponentTeam = input("Enter the opponent team abbreviation (e.g., BOS): ").strip().upper()
    home_input = input("Is your team playing at home? [y/n]: ").strip().lower()
    is_home = True if home_input in ['y', 'yes'] else False

    reg_mse, pred_margin, clf_acc, win_prob = teamPredictionCombinedSimplified(team, opponentTeam, teamSeasonData, teamSchedule, is_home)
    print("\nFinal Combined Output:")
    print(f"Regression MSE: {reg_mse:.2f}")
    print(f"Predicted Point Margin: {pred_margin:.2f}")
    print(f"Classification Accuracy: {clf_acc:.2f}")
    print(f"Win Probability: {win_prob:.2f}")

"""## Generate Scouting Report for Player"""

def playerPredictionWithRegressionAllStats(player, opponentTeam):
  stats = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']
  report = []
  for stat in stats:
    report.append(playerPredictionWithRegression(player, opponentTeam, stat))
  return report

def generatePlayerScoutingReport(player, opponentTeam):
    report = playerPredictionWithRegressionAllStats(player, opponentTeam)

    genai.configure(api_key="AIzaSyAfUkLbGH2Dk3mzM-npUlPDf1zMvpN9ceM")

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""I want you to write a scouting report for me on a player based on a predictive model for a player given the team they play against. My model takes the following parameters like player name, opposing team, and desired stat, and will output an expected stat.

        Here's the output of my model:
        {report}

        Now write the report for me. Don't use bullet points, don't try to add new model parameters or python code. I just want a scouting report. Include a detailed section on strengths and weaknesses.

        Follow this template:
        Scouting Report for {player}:

        Strengths:

        Weaknesses:

        How To Defend:

    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error generating scouting report."

"""##Generate Scouting Report for Team"""

def teamPredictionWithRegressionAllStats(team, opponentTeam):
  players = playerSeasonStats[playerSeasonStats['TEAM_ABBREVIATION'] == team].copy()["PLAYER_NAME"].tolist()
  print(players)
  reportTeam = []
  for player in players:
    reportTeam.append(playerPredictionWithRegressionAllStats(player, opponentTeam))
  return reportTeam


def generateTeamScoutingReport(team, opponentTeam):
    reportTeam = teamPredictionWithRegressionAllStats(team, opponentTeam)

    genai.configure(api_key="AIzaSyAfUkLbGH2Dk3mzM-npUlPDf1zMvpN9ceM")

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt=f"""I want you to write a scouting report for a team based on a predictive model for each of the team's players given the team they play against. My model takes the following parameters like player name, opposing team, and desired stat, and will output an expected stat.

        Here's the output of my model:
        {reportTeam}

        Now write the report for me. Don't use bullet points, don't try to add new model parameters or python code. I just want a scouting report. Include a detailed section on strengths and weaknesses.

        Follow this template:
        Scouting Report for {team}:

        Strengths:

        Weaknesses:

        How To Defend:

    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error generating scouting report."