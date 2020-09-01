import joblib
import pandas as pd
import numpy as np
import pytz
from flask_cors import CORS
from flask import Flask
from flask import request
import streamlit as st
import ipdb

from beatthebookies.gcp import download_model

app = Flask(__name__)
CORS(app)

PATH_TO_MODEL = "model.joblib"

COLS = ['H_ATT', 'A_ATT', 'H_MID', 'A_MID', 'H_DEF', 'A_DEF', 'H_OVR', 'A_OVR',
        'home_t_total_wins','away_t_total_wins', 'stage','home_t_total_goals','away_t_total_goals',
        'home_t_home_goals','home_t_home_goals_against','away_t_away_goals','away_t_away_goals_against',
        'home_t_prev_home_matches', 'away_t_prev_away_matches', 'home_t_total_shots', 'home_t_total_shots_ot',
        'away_t_total_shots', 'away_t_total_shots_ot', 'home_t_total_goals_against','away_t_total_goals_against', 'WHH',
        'WHA', 'WHD', 'home_w', 'away_w', 'draw', 'winning_odds']


def format_input(input):
    formated_input = {}
    for col in COLS:
        formated_input[col] = float(input[col])
    # formated_input = {
    #     'H_ATT': float(input['H_ATT']), 'A_ATT': float(input['A_ATT']), 'H_MID': float(input['H_MID']), 'A_MID': float(input['A_MID']),
    #     'H_DEF': float(input['H_DEF']), 'A_DEF': float(input['A_DEF']), 'H_OVR': float(input['H_OVR']), 'A_OVR': float(input['A_OVR']),
    #     'home_t_total_wins': float(input['home_t_total_wins']),'away_t_total_wins': float(input['away_t_total_wins']), 'stage': float(input['stage']),
    #     'home_t_total_goals': float(input['home_t_total_goals']),'away_t_total_goals': float(input['away_t_total_goals']),
    #     'home_t_home_goals': float(input['home_t_home_goals']),'home_t_home_goals_against': float(input['home_t_home_goals_against']),
    #     'away_t_away_goals': float(input['away_t_away_goals']),'away_t_away_goals_against': float(input['away_t_away_goals_against']),
    #     'home_t_prev_home_matches': float(input['home_t_prev_home_matches']), 'away_t_prev_away_matches': float(input['away_t_prev_away_matches']),
    #     'home_t_total_shots': float(input['home_t_total_shots']), 'home_t_total_shots_ot': float(input['home_t_total_shots_ot']),
    #     'away_t_total_shots': float(input['away_t_total_shots']), 'away_t_total_shots_ot': float(input['away_t_total_shots_ot']),
    #     'home_t_total_goals_against': float(input['home_t_total_goals_against']),'away_t_total_goals_against': float(input['away_t_total_goals_against'])}
    return formated_input


pipeline_def = {'pipeline': joblib.load(PATH_TO_MODEL),
                'from_gcp': False}


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

# st.markdown("**Beat the Bookies**")


# @st.cache
# def read_data():
#     df = get_data(season='2019/2020', league='EPL')
#     return df[df['season'] == season]

# def format_input(home_team, away_team, stage):
#     # names entered could be slightly different variants so maybe a drop down table of suggested team names ?
#     pass


# def main():
#     analysis = st.sidebar.selectbox("Choose prediction type", ["Match Result", "Match Score"])
#     if analysis == 'Match Result':
#         pipeline = joblib.load('data/modelclf.joblib')
#         st.header("Beat the Bookies, pick the winner")
#         # input from user with arbitary default team name suggestions and game week
#         stage = st.text_input('Game Week', 1)
#         home_team = st.text_input('Home Team', 'Arsenal')
#         away_team = st.text_input('Away Team', 'Man City')

#         # missing what to put in the prediction t
#         prediction = pipeline.predict()
#         st.write(f"Looks like your best bet is {prediction[0]}")
#         st.markdown("**Place your bets with your favourite Bookmaker**")
#         st.markdown("Please gamble responsibly, for more information visit https://www.begambleaware.org")

#     if analysis == "Match Score":
#         pipeline = joblib.load('data/model.joblib')
#         st.header("Beat the Bookies, pick the winner")
#         # input from user with arbitary default team name suggestions and game week
#         stage = st.text_input('Game Week', 1)
#         home_team = st.text_input('Home Team', 'Arsenal')
#         away_team = st.text_input('Away Team', 'Man City')

#         # missing what to put in the prediction t
#         prediction = pipeline.predict()
#         st.write(f"Looks like your best bet is {prediction[0]}")
#         st.markdown("**Place your bets with your favourite Bookmaker**")
#         st.markdown("Please gamble responsibly, for more information visit https://www.begambleaware.org")


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
    # main()
