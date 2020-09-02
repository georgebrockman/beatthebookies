import joblib
import pandas as pd
import numpy as np
import pytz
import ipdb
import streamlit as st

from flask_cors import CORS
from flask import Flask
from flask import request
from random import randint, uniform


from beatthebookies.gcp import download_model
from beatthebookies.params import BUCKET_NAME, BUCKET_PREDICT_DATA_PATH

app = Flask(__name__)
CORS(app)

PATH_TO_MODEL = "beatthebookies/data/model.joblib"
# PATH_TO_MODEL = "gs://{}/{}".format(BUCKET_NAME, BUCKET_PREDICT_DATA_PATH)
FIFA_FILE = "gs://{}/{}".format(BUCKET_NAME, BUCKET_PREDICT_DATA_PATH)

DF = pd.read_csv(FIFA_FILE)
TEAMS = DF['Team']

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
    formated_input['H_ATT'] = float(input['home_team']['ATT'].values[0])
    formated_input['A_ATT'] = float(input['away_team']['ATT'].values[0])
    formated_input['H_MID'] = float(input['home_team']['MID'].values[0])
    formated_input['A_MID'] = float(input['away_team']['MID'].values[0])
    formated_input['H_DEF'] = float(input['home_team']['DEF'].values[0])
    formated_input['A_DEF'] = float(input['away_team']['DEF'].values[0])
    formated_input['H_OVR'] = float(input['home_team']['OVR'].values[0])
    formated_input['A_OVR'] = float(input['away_team']['OVR'].values[0])

    print(formated_input)
    return formated_input


# pipeline_def = {'pipeline': joblib.load(PATH_TO_MODEL),
#                 'from_gcp': False}


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


def main():
    analysis = st.sidebar.selectbox("Choose prediction type", ["2020/2021", "2019/2020"])

    if analysis == '2020/2021':
        pipeline = joblib.load(PATH_TO_MODEL)
        st.header("Beat the Bookies")
        # input from user with arbitary default team name suggestions and game week
        home_team = st.selectbox('Home Team', TEAMS)
        away_team = st.selectbox('Away Team', TEAMS)
        if home_team != away_team:
            st.write('You selected:')
            home_stats = DF[DF['Team'] == home_team]
            away_stats = DF[DF['Team'] == away_team]
            df = pd.concat([home_stats, away_stats], axis=0)
            st.table(df.drop(columns='League').set_index('Team').style.highlight_max(axis=0))
            odds = pd.DataFrame({'Home Win': round(uniform(1,2.5),2), 'Away Win': round(uniform(1,10),2), 'Draw': round(uniform(2.5,4),2) }, index=[0])
            st.table(odds.assign(hack='William Hill Odds').set_index('hack'))

            to_predict = {'home_team': home_stats, 'away_team': away_stats}
            to_predict = [to_predict]
            to_predict = [format_input(team) for team in to_predict]
            X = pd.DataFrame(to_predict)
            X = X[COLS]
            prediction = pipeline.predict(X[COLS])
            if prediction[0] == 1:
                st.header(f"Looks like your best bet is the hometeam")
            else:
                st.header(f"Could go either way, stear clear on this one")
            st.markdown("**Place your bets with your favourite Bookmaker**")
        else:
            st.markdown("**Pick an Opponent**")
            st.markdown("And Remember, please gamble responsibly, for more information visit https://www.begambleaware.org")

    if analysis == "Model Results":
        pipeline = joblib.load('data/model.joblib')
        st.header("Beat the Bookies, pick the winner")
        # input from user with arbitary default team name suggestions and game week
        stage = st.text_input('Game Week', 1)
        home_team = st.selectbox('Home Team', TEAMS)
        away_team = st.selectbox('Away Team', TEAMS)

        # missing what to put in the prediction t
        prediction = pipeline.predict(X)
        if prediction[0] == 1:
            st.write(f"##Looks like your best bet is the underdog##")
        else:
            st.write(f"##Stear Clear##")

        st.markdown("**Place your bets with your favourite Bookmaker**")
        st.markdown("Please gamble responsibly, for more information visit https://www.begambleaware.org")


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8080, debug=True)
    main()
