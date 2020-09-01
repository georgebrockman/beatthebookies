import joblib
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
from flask_cors import ColumnTransformer
import streamlit as st

from beatthebookies.gcp import download_model

app = Flask(__name__)
CORS(app)

PATH_TO_MODEL = "model.joblib"

st.markdown("**Beat the Bookies**")

@st.cache
def read_data():
    df = get_data(season='2019/2020', league='EPL')
    return df[df['season'] == season]

def format_input(home_team, away_team, stage):
    # names entered could be slightly different variants so maybe a drop down table of suggested team names ?
    pass


def main():
    analysis = st.sidebar.selectbox("Choose prediction type", ["Match Result", "Match Score"])
    if analysis == 'Match Result':
        pipeline = joblib.load('data/modelclf.joblib')
        st.header("Beat the Bookies, pick the winner")
        # input from user with arbitary default team name suggestions and game week
        stage = st.text_input('Game Week', 1)
        home_team = st.text_input('Home Team', 'Arsenal')
        away_team = st.text_input('Away Team', 'Man City')

        # missing what to put in the prediction t
        prediction = pipeline.predict()
        st.write(f"Looks like your best bet is {prediction[0]}")
        st.markdown("**Place your bets with your favourite Bookmaker**")
        st.markdown("Please gamble responsibly, for more information visit https://www.begambleaware.org")

    if analysis == "Match Score":
        pipeline = joblib.load('data/model.joblib')
        st.header("Beat the Bookies, pick the winner")
        # input from user with arbitary default team name suggestions and game week
        stage = st.text_input('Game Week', 1)
        home_team = st.text_input('Home Team', 'Arsenal')
        away_team = st.text_input('Away Team', 'Man City')

        # missing what to put in the prediction t
        prediction = pipeline.predict()
        st.write(f"Looks like your best bet is {prediction[0]}")
        st.markdown("**Place your bets with your favourite Bookmaker**")
        st.markdown("Please gamble responsibly, for more information visit https://www.begambleaware.org")


if __name__ == "__main__":
    main()
