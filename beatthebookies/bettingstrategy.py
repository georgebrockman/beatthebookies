import pandas as pd
import numpy as np
import sqlite3

from beatthebookies.data import get_betting_data

def simple_betting_profits(df, bet=10):
    """
    function returns the cumulative profits/loss from a season of following a consistent, simple betting strategy.
    """
    # set defaults profit equal to bet size
    # create new column for fav strategy
    df['fav_profit'] = -bet
    # update profit column
    df.loc[(df[['LBH','LBD','LBA']].min(axis=1) == df['LBH']) & (df['home_w'] == 1), 'fav_profit'] = df['LBH'] * bet
    df.loc[(df[['LBH','LBD','LBA']].min(axis=1) == df['LBD']) & (df['draw'] == 1), 'fav_profit'] = df['LBD'] * bet
    df.loc[(df[['LBH','LBD','LBA']].min(axis=1) == df['LBA']) & (df['away_w'] == 1), 'fav_profit'] = df['LBA'] * bet

    # set defaults profit equal to bet size
    # create new column for underdog strategy
    df['dog_profit'] = -bet
    # update profit column
    df.loc[(df[['LBH','LBD','LBA']].max(axis=1) == df['LBH']) & (df['home_w'] == 1), 'dog_profit'] = df['LBH'] * bet
    df.loc[(df[['LBH','LBD','LBA']].max(axis=1) == df['LBD']) & (df['draw'] == 1), 'dog_profit'] = df['LBD'] * bet
    df.loc[(df[['LBH','LBD','LBA']].max(axis=1) == df['LBA']) & (df['away_w'] == 1), 'dog_profit'] = df['LBA'] * bet

        # create new column for home team method
    df['home_profit'] = -bet
    df.loc[(df['home_w'] == 1), 'home_profit'] = df['LBH'] * bet
    # create new column for draw tactic
    df['draw_profit'] = -bet
    df.loc[(df['draw'] == 1), 'draw_profit'] = df['LBD'] * bet
    # create new column for betting on the away team
    df['away_profit'] = -bet
    df.loc[(df['away_w'] == 1), 'away_profit'] = df['LBA'] * bet

    fav_profit_total = df['fav_profit'].sum()
    dog_profit_total = df['dog_profit'].sum()
    home_profit_total = df['home_profit'].sum()
    draw_profit_total = df['draw_profit'].sum()
    away_profit_total = df['away_profit'].sum()

    return fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total

def model_profits(df, bet=10):
    pass
