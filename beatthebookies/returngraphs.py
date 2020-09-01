# could make this file more concise by cleaning up the betting strategies file and just importing from there - up to you guys but not an essential fix!
# line plot for the models projection is commented out for now

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style

from beatthebookies.csvcompiler import get_csv_data

df = get_csv_data()

def profit_loss_graph(df=df, season='2019/2020'):
    def graph_data(df=df, season='2019/2020', bet=10):
        # create new DataFrame limited to only desired season
        pl_season = df[df['season'] == season]
        # create empty DataFrame with each betting strategy
        strategy = pd.DataFrame(columns=['fav_profit', 'underdog_profit', 'home_profit','draw_profit', 'away_profit', 'model_profit', 'game_week'])
        # fav win strategy
        pl_season['fav_profit'] = -bet
        pl_season.loc[(pl_season[['WHH','WHD','WHA']].min(axis=1) == pl_season['WHH']) & (pl_season['home_w'] == 1), 'fav_profit'] = (pl_season['WHH'] * bet) - bet
        pl_season.loc[(pl_season[['WHH','WHD','WHA']].min(axis=1) == pl_season['WHD']) & (pl_season['draw'] == 1), 'fav_profit'] = (pl_season['WHD'] * bet) - bet
        pl_season.loc[(pl_season[['WHH','WHD','WHA']].min(axis=1) == pl_season['WHA']) & (pl_season['away_w'] == 1), 'fav_profit'] = (pl_season['WHA'] * bet) - bet
        # under dog
        pl_season['dog_profit'] = -bet
        pl_season.loc[(pl_season[['WHH','WHD','WHA']].max(axis=1) == pl_season['WHH']) & (pl_season['home_w'] == 1), 'dog_profit'] = (pl_season['WHH'] * bet) - bet
        pl_season.loc[(pl_season[['WHH','WHD','WHA']].max(axis=1) == pl_season['WHD']) & (pl_season['draw'] == 1), 'dog_profit'] = (pl_season['WHD'] * bet) - bet
        pl_season.loc[(pl_season[['WHH','WHD','WHA']].max(axis=1) == pl_season['WHA']) & (pl_season['away_w'] == 1), 'dog_profit'] = (pl_season['WHA'] * bet) - bet
        # home, draw, away
        pl_season['home_profit'] = -bet
        pl_season.loc[(pl_season['home_w'] == 1), 'home_profit'] = (pl_season['WHH'] * bet) - bet
        pl_season['draw_profit'] = -bet
        pl_season.loc[(pl_season['draw'] == 1), 'draw_profit'] = (pl_season['WHD'] * bet) - bet
        pl_season['away_profit'] = -bet
        pl_season.loc[(pl_season['away_w'] == 1), 'away_profit'] = (pl_season['WHA'] * bet) - bet
        # model
        pl_season.model_profit = pass
        # update strategy DataFrame
        strategy.fav_profit =  pl_season.fav_profit
        strategy.underdog_profit =  pl_season.dog_profit
        strategy.home_profit = pl_season.home_profit
        strategy.draw_profit = pl_season.draw_profit
        strategys.away_profit = pl_season.away_profit
        strategy.game_week = pl_season.stage
        # reset index
        strategy.reset_index(drop=True, inplace=True)
        # create a cumulaitve total for each strategy in the season
        strategy.home_profit = strategy.home_profit.cumsum().fillna(0)
        strategy.draw_profit = strategy.draw_profit.cumsum().fillna(0)
        strategy.away_profit = strategy.away_profit.cumsum().fillna(0)
        strategy.fav_profit = strategy.fav_profit.cumsum().fillna(0)
        strategy.underdog_profit = strategy.underdog_profit.cumsum().fillna(0)
        strategy.underdog_profit = strategy.underdog_profit.cumsum().fillna(0)

    def get_graph(df=strategy):
        style.use('seaborn-poster')
        sns.lineplot(x='game_week', y='fav_profit', data=df, label='Back the Favourite', ci=None)
        sns.lineplot(x='game_week', y='underdog_profit', data=df, label='Back the Underdog', ci=None)
        sns.lineplot(x='game_week', y='home_profit', data=df, label='Back the Home Team', ci=None)
        sns.lineplot(x='game_week', y='away_profit', data=df, label='Back the Away Team', ci=None)
        sns.lineplot(x='game_week', y='draw_profit', data=df, label='Back a Draw', ci=None)
        sns.lineplot(x='game_week', y='model_profit', data=df, label='Back a Draw', ci=None)
        # sns.lineplot(x='game_week', y='model_profit', data=profits, label='Model profit', ci=None)
        plt.legend(fontsize='large', title_fontsize='40', loc='upper left')
        plt.ylabel("Profit/Loss")
        plt.xlabel("Game Week")
        plt.title(f"Beat the Bookies Profit/Loss Projection for the {season} Season")

        return plt.show() # double check that this is how you return the graph from a function
