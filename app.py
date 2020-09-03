import joblib
import pandas as pd
import numpy as np
import pytz
import streamlit as st

from flask_cors import CORS
from flask import Flask
from flask import request
from random import randint, uniform


from beatthebookies.gcp import download_model
from beatthebookies.params import BUCKET_NAME, BUCKET_PREDICT_DATA_PATH, MODEL_NAME, MODEL_VERSION
from beatthebookies.bettingstrategy import kelly_prediction

app = Flask(__name__)
CORS(app)

class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


PATH_TO_MODEL = "model.joblib"
FIFA_FILE = 'fifarank21.csv'
SEASON_FILE = '2019_2020.csv'
# GCP_PATH_TO_MODEL = 'models/{}/versions/{}/{}'.format(MODEL_NAME,MODEL_VERSION,'model.joblib')
# GCP_FIFA_FILE = "gs://{}/{}".format(BUCKET_NAME, BUCKET_PREDICT_DATA_PATH)

FIFA_DF = pd.read_csv(FIFA_FILE)
SEASON_DF = pd.read_csv(SEASON_FILE)
TEAMS = FIFA_DF['Team']

COLS = ['H_ATT', 'A_ATT', 'H_MID', 'A_MID', 'H_DEF', 'A_DEF', 'H_OVR', 'A_OVR',
        'home_t_total_wins','away_t_total_wins', 'stage','home_t_total_goals','away_t_total_goals',
        'home_t_home_goals','home_t_home_goals_against','away_t_away_goals','away_t_away_goals_against',
        'home_t_prev_home_matches', 'away_t_prev_away_matches', 'home_t_total_shots', 'home_t_total_shots_ot',
        'away_t_total_shots', 'away_t_total_shots_ot', 'home_t_total_goals_against','away_t_total_goals_against', 'WHH',
        'WHA', 'WHD', 'home_w', 'away_w', 'draw', 'winning_odds']


def format_input(input):
    formated_input = {}
    for col in COLS:
        formated_input[col] = randint(1,20) #float(input[col])
    formated_input['H_ATT'] = float(input['home_rank']['ATT'].values[0])
    formated_input['A_ATT'] = float(input['away_rank']['ATT'].values[0])
    formated_input['H_MID'] = float(input['home_rank']['MID'].values[0])
    formated_input['A_MID'] = float(input['away_rank']['MID'].values[0])
    formated_input['H_DEF'] = float(input['home_rank']['DEF'].values[0])
    formated_input['A_DEF'] = float(input['away_rank']['DEF'].values[0])
    formated_input['H_OVR'] = float(input['home_rank']['OVR'].values[0])
    formated_input['A_OVR'] = float(input['away_rank']['OVR'].values[0])
    if 'home_stats' in input.keys():
        s_dict = input['home_stats'].to_dict(orient='list')
        for k,v in s_dict.items():
            if k in COLS:
                formated_input[k] = v[0]

    if 'away_stats' in input.keys():
        s_dict = input['away_stats'].to_dict(orient='list')
        for k,v in s_dict.items():
            if k in COLS:
                formated_input[k] = v[0]

    if 'game_stats' in input.keys():
        s_dict = input['game_stats'].to_dict(orient='list')
        for k,v in s_dict.items():
            if k in COLS:
                formated_input[k] = v[0]

    return formated_input



@app.route('/')
def index():
    return {'respone': 'OK'}


@app.route('/predict_match', methods=['GET', 'POST'])
def predict_match():
    """
    Expected input
        {32 fields}
    :return: {"predictions": 1 or 0 }
    """
    inputs = request.get_json()
    # ipdb.set_trace()
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs = [format_input(stat) for stat in inputs]
    # Here wee need to convert inputs to dataframe to feed as input to our pipeline
    # Indeed our pipeline expects a dataframe as input
    X = pd.DataFrame(inputs)
    # Here we specify the right column order
    X = X[COLS]
    pipeline = pipeline_def["pipeline"]
    results = pipeline.predict(X)
    # results = [round(float(r), 3) for r in results]
    return {"predictions": str(results[0])}


@app.route('/set_model', methods=['GET', 'POST'])
def set_model():
    inputs = request.get_json()
    model_dir = inputs["model_directory"]
    pipeline_def["pipeline"] = download_model(model_directory=model_dir, rm=True)
    pipeline_def["from_gcp"] = True
    return {"reponse": f"correctly got model from {model_dir} directory on GCP"}

st.markdown("**Beat the Bookies**")


# @st.cache
# def read_data():
#     df = get_data(season='2019/2020', league='EPL')
#     return df[df['season'] == season]

# def format_input(home_team, away_team, stage):
#     # names entered could be slightly different variants so maybe a drop down table of suggested team names ?
#     pass

def generate_game():
    low_stats = ['home_t_total_wins', 'home_t_total_losses', 'home_t_total_draws', 'home_t_prev_home_matches', 'away_t_total_wins',
                  'away_t_total_losses', 'away_t_total_draws', 'away_t_prev_away_matches', 'stage']

    mid_stats = ['home_t_total_goals', 'home_t_home_goals', 'home_t_home_goals_against','home_t_total_goals_against',
                'away_t_total_goals', 'away_t_away_goals', 'away_t_away_goals_against','away_t_total_goals_against']
    high_stats = ['away_t_total_shots_ot','home_t_total_shots_ot']
    highest_stats = [ 'home_t_total_shots', 'away_t_total_shots']
    game = {}
    for stat in low_stats:
        game[stat] = randint(1,5)
    for stat in mid_stats:
        game[stat] = randint(5,15)
    for stat in high_stats:
        game[stat] = randint(30,40)
    for stat in highest_stats:
        game[stat] = randint(60,80)
    game_df = pd.DataFrame(game, index=[0])
    return game_df


def main():
    analysis = st.sidebar.selectbox("Choose prediction type", ["2020/2021", "2019/2020"])

    if analysis == '2020/2021':
        pipeline = joblib.load(PATH_TO_MODEL)
        st.header("Beat the Bookies")
        # input from user with arbitary default team name suggestions and game week
        home_team = st.selectbox('Home Team', TEAMS)
        away_team = st.selectbox('Away Team', TEAMS)
        bet = st.number_input('Max Bet Per Game')
        if home_team != away_team:
            st.write('You selected:')

            st.write('**Fifa Rankings**')
            home_ranks = FIFA_DF[FIFA_DF['Team'] == home_team]
            away_ranks = FIFA_DF[FIFA_DF['Team'] == away_team]
            df = pd.concat([home_ranks, away_ranks], axis=0)
            st.table(df.drop(columns='League').set_index('Team').style.highlight_max(axis=0))
            game = generate_game()
            home_stats = game[['home_t_total_wins', 'home_t_total_losses', 'home_t_total_draws', 'home_t_total_goals', 'home_t_home_goals', 'home_t_home_goals_against',
                              'home_t_prev_home_matches', 'home_t_total_shots', 'home_t_total_shots_ot', 'home_t_total_goals_against']]
            away_stats = game[['away_t_total_wins', 'away_t_total_losses', 'away_t_total_draws', 'away_t_total_goals', 'away_t_away_goals', 'away_t_away_goals_against',
                              'away_t_prev_away_matches', 'away_t_total_shots', 'away_t_total_shots_ot', 'away_t_total_goals_against']]
            away_stats['Team'] = away_team
            home_stats['Team'] = home_team

            home_show =  home_stats[['Team','home_t_total_wins', 'home_t_total_losses', 'home_t_total_draws', 'home_t_total_goals', 'home_t_total_goals_against', 'home_t_total_shots', 'home_t_total_shots_ot']]
            away_show =  away_stats[['Team','away_t_total_wins', 'away_t_total_losses', 'away_t_total_draws', 'away_t_total_goals', 'away_t_total_goals_against', 'away_t_total_shots', 'away_t_total_shots_ot']]
            cols = ['Team', 'Wins', 'Losses', 'Draws', 'Goals', 'Goals Against', 'Shots', 'Shots On Target']
            home_show.columns = cols
            away_show.columns = cols

            st.write('**Team Stats**')
            stat_df = pd.concat([home_show, away_show], axis=0)
            st.table(stat_df.set_index('Team'))

            st.write('**Game Odds**')
            odds = pd.DataFrame({'Home Win': round(uniform(1.3,3.5),2), 'Away Win': round(uniform(1.5,8),2), 'Draw': round(uniform(2.5,4),2) }, index=[0])
            st.table(odds.assign(hack='William Hill Odds').set_index('hack'))
            odds['stage'] = game['stage']
            game_stats = odds[['Home Win', 'Away Win', 'Draw', 'stage']]
            game_stats.columns = ['WHH', 'WHA', 'WHD', 'stage']
            to_predict = {'home_rank': home_ranks, 'away_rank': away_ranks, 'home_stats': home_stats, 'away_stats': away_stats, 'game_stats': game_stats}
            to_predict = [to_predict]
            to_predict = [format_input(team) for team in to_predict]
            X = pd.DataFrame(to_predict)
            X = X[COLS]
            prediction = pipeline.predict(X[COLS])[0]
            prediction_proba = pipeline.predict_proba(X[COLS])[0][1]
            kelly = kelly_prediction(prediction_proba, odds['Home Win'].values[0], bet)
            if prediction == 1:
                st.subheader(f"Your best bet is on the hometeam with {round(prediction_proba * 100,2)}% certainty")
                st.subheader(f"The kelly principle suggests a bet of £{kelly}")
            else:
                st.subheader(f"Not looking great for the hometeam, only a {round(prediction_proba * 100,2)}% chance of a win")
                st.subheader(f"The kelly principle suggests a bet of £{kelly}")
            st.markdown("**Place your bets with your favourite Bookmaker**")
        else:
            st.markdown("**Pick an Opponent**")
            st.markdown("And Remember, please gamble responsibly, for more information visit https://www.begambleaware.org")

    if analysis == "2019/2020":

        pipeline = joblib.load('model.joblib')
        # input from user with arbitary default team name suggestions and game week
        st.markdown("""<a id="top"></a>""",unsafe_allow_html=True)
        stage = st.number_input('Game Week', min_value=1, max_value=39)
        bet = st.number_input('Max Bet Per Game')
        toc = Toc()
        href = f"""<a href="#top"> Back to top </a>"""
        st.title("This Weeks Games")
        toc.placeholder()

        matchups = SEASON_DF[SEASON_DF['stage'] == stage]
        profits = []
        for num in range(10):
            game = matchups[num:num+1]
            ht = game.home_team.values[0]
            toc.header(f'{num + 1}. {ht} Match')
            home_ranks = game[['home_team', 'H_ATT', 'H_MID', 'H_DEF', 'H_OVR']]
            home_ranks.columns = ['Team', 'ATT', 'MID', 'DEF', 'OVR']
            away_ranks = game[['away_team', 'A_ATT', 'A_MID', 'A_DEF', 'A_OVR']]
            away_ranks.columns = ['Team', 'ATT', 'MID', 'DEF', 'OVR']
            df = pd.concat([home_ranks, away_ranks], axis=0)
            st.write('**Fifa Rankings**')
            st.table(df.set_index('Team').style.highlight_max(axis=0))
            home_stats = game[['home_t_total_wins', 'home_t_total_losses', 'home_t_total_draws', 'home_t_total_goals', 'home_t_home_goals', 'home_t_home_goals_against',
                              'home_t_prev_home_matches', 'home_t_total_shots', 'home_t_total_shots_ot', 'home_t_total_goals_against']]
            away_stats = game[['away_t_total_wins', 'away_t_total_losses', 'away_t_total_draws', 'away_t_total_goals', 'away_t_away_goals', 'away_t_away_goals_against',
                              'away_t_prev_away_matches', 'away_t_total_shots', 'away_t_total_shots_ot', 'away_t_total_goals_against']]
            away_stats['Team'] = game['away_team'].copy()
            home_stats['Team'] = game['home_team'].copy()
            home_show =  home_stats[['Team','home_t_total_wins', 'home_t_total_losses', 'home_t_total_draws', 'home_t_total_goals', 'home_t_total_goals_against', 'home_t_total_shots', 'home_t_total_shots_ot']]
            away_show =  away_stats[['Team','away_t_total_wins', 'away_t_total_losses', 'away_t_total_draws', 'away_t_total_goals', 'away_t_total_goals_against', 'away_t_total_shots', 'away_t_total_shots_ot']]
            cols = ['Team', 'Wins', 'Losses', 'Draws', 'Goals', 'Goals Against', 'Shots', 'Shots On Target']
            home_show.columns = cols
            away_show.columns = cols
            st.write('**Team Stats**')
            stat_df = pd.concat([home_show, away_show], axis=0)
            st.table(stat_df.set_index('Team'))
            odds = game[['WHH', 'WHA', 'WHD', 'FTR']]
            odds.columns = ['Home Win', 'Away Win', 'Draw', 'Results']
            st.write('**Game Odds**')
            st.table(odds.assign(hack='William Hill Odds').set_index('hack'))
            game_stats = game[['WHH','WHA', 'WHD', 'stage']]
            to_predict = {'home_rank': home_ranks, 'away_rank': away_ranks, 'home_stats': home_stats, 'away_stats': away_stats, 'game_stats': game_stats}
            to_predict = [to_predict]
            to_predict = [format_input(team) for team in to_predict]
            X = pd.DataFrame(to_predict)
            X = X[COLS]
            prediction = pipeline.predict(X[COLS])[0]
            prediction_proba = pipeline.predict_proba(X[COLS])[0][1]
            kelly = kelly_prediction(prediction_proba, game.WHH.values[0], bet)
            if prediction == 1:
                st.subheader(f"Your best bet is on the hometeam with {round(prediction_proba * 100,2)}% certainty")
                st.subheader(f"The kelly principle suggests a bet of £{kelly}")
                if (odds['Results'].values[0] == 'H'):
                    profit = round((kelly * game.WHH.values[0]) - kelly, 2)
                    collection = round(kelly * game.WHH.values[0], 2)
                    profits.append(collection)
                    st.write(f'Collect your £{collection}')
                else:
                    profit = -kelly
                    profits.append(profit)
                    st.write("Nobody bats 1000")
            else:
                st.subheader(f"Not looking great for the hometeam, only a {round(prediction_proba * 100,2)}% chance of a win")
                st.subheader(f"The kelly principle suggests a bet of £{kelly}")
                if (odds['Results'].values[0] == 'H'):
                    st.write('Well, better safe than sorry')
                else:
                    st.write("Nailed it")
            st.markdown(href,unsafe_allow_html=True)
            st.write('___')



        weekly_profit = round(sum(profits),2)
        toc.header(f'Weekly Results')
        st.write(f'**Total collections for the week were £{weekly_profit}**')
        if weekly_profit > 0:
            st.balloons()

        st.markdown("Please gamble responsibly, for more information visit https://www.begambleaware.org")
        st.markdown(href,unsafe_allow_html=True)
        toc.generate()


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8080, debug=True)
    main()
