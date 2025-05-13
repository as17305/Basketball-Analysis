from flask import Flask, render_template, request
import os
from backend import *
from pyngrok import ngrok, conf

# Set ngrok auth token directly
os.environ['NGROK_AUTH_TOKEN'] = "2wyA3n667aUo8jySIwHjVw6aE9W_4hoxecZga6mGKyRDCEUzb"

app = Flask(__name__)

def get_players_and_teams():
    players = playerSeasonStats["PLAYER_NAME"].tolist()
    teams = [
        ('ATL', 'Atlanta Hawks'),
        ('BOS', 'Boston Celtics'),
        ('BKN', 'Brooklyn Nets'),
        ('CHA', 'Charlotte Hornets'),
        ('CHI', 'Chicago Bulls'),
        ('CLE', 'Cleveland Cavaliers'),
        ('DAL', 'Dallas Mavericks'),
        ('DEN', 'Denver Nuggets'),
        ('DET', 'Detroit Pistons'),
        ('GSW', 'Golden State Warriors'),
        ('HOU', 'Houston Rockets'),
        ('IND', 'Indiana Pacers'),
        ('LAC', 'Los Angeles Clippers'),
        ('LAL', 'Los Angeles Lakers'),
        ('MEM', 'Memphis Grizzlies'),
        ('MIA', 'Miami Heat'),
        ('MIL', 'Milwaukee Bucks'),
        ('MIN', 'Minnesota Timberwolves'),
        ('NOP', 'New Orleans Pelicans'),
        ('NYK', 'New York Knicks'),
        ('OKC', 'Oklahoma City Thunder'),
        ('ORL', 'Orlando Magic'),
        ('PHI', 'Philadelphia 76ers'),
        ('PHX', 'Phoenix Suns'),
        ('POR', 'Portland Trail Blazers'),
        ('SAC', 'Sacramento Kings'),
        ('SAS', 'San Antonio Spurs'),
        ('TOR', 'Toronto Raptors'),
        ('UTA', 'Utah Jazz'),
        ('WAS', 'Washington Wizards')
    ]
    return players, teams

def generate_player_predictions(player, team_code, teams):
    team_name = next((name for code, name in teams if code == team_code), '')

    def extract_value(pred_str):
        try:
            return float(pred_str.split(":")[-1].strip())
        except:
            return None

    pred_pts = playerPredictionWithRegression(player, team_code, 'PTS')
    pred_reb = playerPredictionWithRegression(player, team_code, 'REB')
    pred_ast = playerPredictionWithRegression(player, team_code, 'AST')

    return {
        "player_name": player,
        "opponent_team": team_code,
        "opponent_name": team_name,
        "predicted_points": extract_value(pred_pts),
        "predicted_rebounds": extract_value(pred_reb),
        "predicted_assists": extract_value(pred_ast),
        "analysis": f"{player} is projected to score {extract_value(pred_pts)} PTS, grab {extract_value(pred_reb)} REB, and dish {extract_value(pred_ast)} AST against the {team_name} based on model predictions."
    }

def generate_player_visuals(player, stat):
    performance_chart = graphTimeSeriesPlotAllStats(player)

    if stat:
        scatter = graphPlayerScatterPlot(player, stat)
        density = graphPlayerDensityPlot(player, stat)
        time_series = graphTimeSeriesPlot(player, stat)
        return performance_chart, scatter, density, time_series
    else:
        return performance_chart, None, None, None

def compute_opponent_ppg(teamSchedule):
    team_points_allowed = {}

    for _, row in teamSchedule.iterrows():
        try:
            away_pts, home_pts = map(int, row['score'].split('-'))
            home_team = row['hometeam']
            away_team = row['awayteam']

            team_points_allowed.setdefault(home_team, []).append(away_pts)
            team_points_allowed.setdefault(away_team, []).append(home_pts)
        except Exception as e:
            continue

    opp_ppg = {team: sum(pts) / len(pts) for team, pts in team_points_allowed.items()}
    return opp_ppg

def compute_team_wins_losses(teamSchedule):
    win_loss = {}

    for _, row in teamSchedule.iterrows():
        try:
            home_team = row['hometeam']
            away_team = row['awayteam']
            home_score, away_score = map(int, row['score'].split('-'))

            if home_score > away_score:
                win_loss.setdefault(home_team, {"wins": 0, "losses": 0})["wins"] += 1
                win_loss.setdefault(away_team, {"wins": 0, "losses": 0})["losses"] += 1
            else:
                win_loss.setdefault(home_team, {"wins": 0, "losses": 0})["losses"] += 1
                win_loss.setdefault(away_team, {"wins": 0, "losses": 0})["wins"] += 1
        except:
            continue

    return win_loss

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/player_scouting', methods=['GET'])
def player_scouting_get():
    players, teams = get_players_and_teams()
    return render_template(
        'player_scouting.html',
        players=players,
        teams=teams,
        selected_player=None,
        selected_team=None,
        scouting_report=None,
        performance_chart=None,
        player_scatter=None,
        player_density=None,
        time_series=None,
        stat=None
    )

@app.route('/player_scouting', methods=['POST'])
def player_scouting_post():
    stat = request.form.get('stat')
    players, teams = get_players_and_teams()

    selected_player = request.form.get('player')
    selected_team = request.form.get('team')

    scouting_report = None
    performance_chart = None
    player_scatter = None
    player_density = None
    time_series = None

    if selected_player and selected_team:
        scouting_report = generate_player_predictions(selected_player, selected_team, teams)
        performance_chart, player_scatter, player_density, time_series = generate_player_visuals(selected_player, stat)

    gemini_report = None
    try:
        gemini_report = generatePlayerScoutingReport(selected_player, selected_team)
    except Exception as e:
        print("Gemini error:", e)

    return render_template(
        'player_scouting.html',
        players=players,
        teams=teams,
        selected_player=selected_player,
        selected_team=selected_team,
        scouting_report=scouting_report,
        performance_chart=performance_chart,
        player_scatter=player_scatter,
        player_density=player_density,
        time_series=time_series,
        stat=stat,
        gemini_report=gemini_report
    )

@app.route('/team_analysis', methods=['GET'])
def team_analysis_get():
    _, teams = get_players_and_teams()
    team_data = teamSeasonData.to_dict(orient='records')

    return render_template(
        'team_analysis.html',
        teams=teams,
        team_data=team_data,
        selected_team=None,
        selected_opponent=None,
        team_stats=None,
        prediction_result=None,
        gemini_report=None,
        team_graphs={}
    )


@app.route('/team_analysis', methods=['POST'])
def team_analysis_post():
    _, teams = get_players_and_teams()
    team_df = teamSeasonData.copy()
    team_data = team_df.to_dict(orient='records')

    selected_team = request.form.get('team')
    selected_opponent = request.form.get('opponent')
    metrics = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'Turnovers']

    team_stats = None
    prediction_result = None
    gemini_report = None
    team_graphs = {}

    if selected_team:
        team_stats = next((team for team in team_data if team['Team'] == selected_team), None)
        
        if team_stats:  # Add this check to ensure team_stats exists
            team_stats['ppg'] = team_stats['PTS']
            team_stats['fg_pct'] = team_stats['FGPercent']
            team_stats['three_pct'] = team_stats['3PPercent']
            team_stats['oppg'] = compute_opponent_ppg(teamSchedule).get(selected_team, 0)
            team_stats['reb'] = team_stats['REB']
            team_stats['ast'] = team_stats['AST']
            
            try:
                team_stats['composite_score'] = composite_team_ranking().set_index('Team').at[selected_team, 'Composite']
            except:
                team_stats['composite_score'] = 0
            
            if 'Team' in team_stats:
                team_stats['team_name'] = TEAM_CODE_TO_NAME.get(team_stats['Team'], team_stats['Team'])

            win_loss_map = compute_team_wins_losses(teamSchedule)
            if selected_team in win_loss_map:
                team_stats['wins'] = win_loss_map[selected_team]['wins']
                team_stats['losses'] = win_loss_map[selected_team]['losses']
            else:
                team_stats['wins'] = team_stats['losses'] = 0

            opp_ppg_map = compute_opponent_ppg(teamSchedule)
            team_df['OppPTS'] = team_df['Team'].map(opp_ppg_map)

            league_avg = {
                'ppg': team_df['PTS'].mean(),
                'oppg': team_df['OppPTS'].mean(),
                'fg_pct': team_df['FGPercent'].mean(),
                'three_pct': team_df['3PPercent'].mean(),
                'reb': team_df['REB'].mean(),
                'ast': team_df['AST'].mean()
            }
            team_stats['league_avg'] = league_avg

            for stat, col_name in {
                'ppg': 'PTS',
                'fg_pct': 'FGPercent',
                'three_pct': '3PPercent',
                'reb': 'REB',
                'ast': 'AST'
            }.items():
                sorted_teams = team_df.sort_values(by=col_name, ascending=False)
                team_stats[f'{stat}_rank'] = int(
                    sorted_teams.reset_index(drop=True).index[sorted_teams['Team'] == selected_team].tolist()[0] + 1
                )

            sorted_oppg = team_df.sort_values(by='OppPTS')
            team_stats['oppg_rank'] = int(
                sorted_oppg.reset_index(drop=True).index[sorted_oppg['Team'] == selected_team].tolist()[0] + 1
            )

            try:
                df_radar = composite_team_ranking()

                team_graphs['composite'] = graph_team_composite_ranking()
                team_graphs['scatter'] = graph_team_scatter()
                team_graphs['radar'] = plot_radar_chart(df_radar, selected_team, metrics)
                team_graphs['parallel'] = plot_parallel_coordinates(df_radar, metrics)
                team_graphs['game_performance'] = plot_team_game_performance(selected_team)
                team_graphs['heatmap'] = plot_team_heatmap(teamSeasonData, selected_team)

            except Exception as e:
                print("Error generating graphs:", e)

    if selected_team and selected_opponent and team_stats:  # Ensure team_stats exists
        try:
            _, margin, _, win_prob = teamPredictionCombinedSimplified(
                selected_team, selected_opponent, teamSeasonData, teamSchedule, is_home=True
            )
            prediction_result = {
                'margin': round(margin, 2),
                'win_prob': round(win_prob * 100, 1)
            }
            gemini_report = generateTeamScoutingReport(selected_team, selected_opponent)
        except Exception as e:
            print("Prediction error:", e)

    return render_template(
        'team_analysis.html',
        teams=teams,
        team_data=team_data,
        selected_team=selected_team,
        selected_opponent=selected_opponent,
        team_stats=team_stats,
        prediction_result=prediction_result,
        gemini_report=gemini_report,
        team_graphs=team_graphs
    )

@app.route('/player_ranking', methods=['GET', 'POST'])
def player_ranking():
    stats = request.args.getlist('stats')
    if not stats:
        stats = ['PTS', 'REB', 'AST']
    df = composite_ranking(stats)
    top10 = df.head(10)
    players = top10.to_dict(orient='records')
    return render_template(
        'player_ranking.html',
        players=players,
        stats=stats
    )

@app.route('/team_ranking')
def team_ranking():
    df = composite_team_ranking()
    chart = None

    try:
        chart = graph_team_composite_ranking()
    except Exception as e:
        print("Error generating chart:", e)

    top_teams = df.head(10).to_dict(orient='records')

    return render_template(
        'team_ranking.html',
        top_teams=top_teams,
        chart=chart
    )

def start_ngrok():
    # Set up a tunnel on port 5000 (Flask's default port)
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000\"")
    return public_url

if __name__ == '__main__':
    public_url = start_ngrok()
    app.run()