import pandas as pd
import numpy as np
import sqlite3


def simple_betting_profits(df, bet=10):
    """
    function returns the cumulative profits/loss from a season of following a consistent, simple betting strategy.
    """
    # set defaults profit equal to bet size
    # create new column for fav strategy
    df['fav_profit'] = -bet
    # update profit column - (odds * bet) - stake
    df.loc[(df[['WHH','WHD','WHA']].min(axis=1) == df['WHH']) & (df['home_w'] == 1), 'fav_profit'] = (df['WHH'] * bet) - bet
    df.loc[(df[['WHH','WHD','WHA']].min(axis=1) == df['WHD']) & (df['draw'] == 1), 'fav_profit'] = (df['WHD'] * bet) - bet
    df.loc[(df[['WHH','WHD','WHA']].min(axis=1) == df['WHA']) & (df['away_w'] == 1), 'fav_profit'] = (df['WHA'] * bet) - bet

    # set defaults profit equal to bet size
    # create new column for underdog strategy
    df['dog_profit'] = -bet
    # update profit column
    df.loc[(df[['WHH','WHD','WHA']].max(axis=1) == df['WHH']) & (df['home_w'] == 1), 'dog_profit'] = (df['WHH'] * bet) - bet
    df.loc[(df[['WHH','WHD','WHA']].max(axis=1) == df['WHD']) & (df['draw'] == 1), 'dog_profit'] = (df['WHD'] * bet) - bet
    df.loc[(df[['WHH','WHD','WHA']].max(axis=1) == df['WHA']) & (df['away_w'] == 1), 'dog_profit'] = (df['WHA'] * bet) - bet

    # create new column for home team method
    df['home_profit'] = -bet
    df.loc[(df['home_w'] == 1), 'home_profit'] = (df['WHH'] * bet) - bet
    # create new column for draw tactic
    df['draw_profit'] = -bet
    df.loc[(df['draw'] == 1), 'draw_profit'] = (df['WHD'] * bet) - bet
    # create new column for betting on the away team
    df['away_profit'] = -bet
    df.loc[(df['away_w'] == 1), 'away_profit'] = (df['WHA'] * bet) - bet

    fav_profit_total = df['fav_profit'].sum()
    dog_profit_total = df['dog_profit'].sum()
    home_profit_total = df['home_profit'].sum()
    draw_profit_total = df['draw_profit'].sum()
    away_profit_total = df['away_profit'].sum()

    return fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total


def correct_bet(x):
    if x['pred'] == 1:
        if x['act'] == 1:
          return abs(x['bet'] * x['winning_odds']) + x['bet']
        return x['bet']
    return 0



def compute_profit(df,y_pred,y_true,bet):
    fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total = simple_betting_profits(df.copy(), bet=bet)
    combined = pd.DataFrame({'pred': y_pred, 'act': y_true, 'winning_odds': df['winning_odds']})
    combined['bet'] = -bet
    combined['profit'] = combined.apply(lambda x: correct_bet(x) , axis=1)
    btb_profit_total = combined['profit'].sum()
    combined = combined.sort_values(by='profit', ascending=False)
    return btb_profit_total, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total









