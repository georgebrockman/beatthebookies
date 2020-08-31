import pandas as pd
import numpy as np
import sqlite3
from utils import compute_overall_scores


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

def model_profits(df, bet=10):
    pass

def correct_single(x):
    if x['pred'] == 1:
        if x['act'] == 1:
          return abs(x['bet'] * x['winning_odds']) + x['bet']
        return x['bet']
    return 0

def correct_multi(x):
    if x['diff'] == 0:
        return abs(x['bet'] * x['winning_odds']) + x['bet']
    return x['bet']

def correct_threshold(x):
    if x['pct'] > x['threshold']:
        if x['act'] == 1:
            return abs(x['bet'] * x['winning_odds']) + x['bet']
        else:
          return x['bet']
    return 0

def round_to(x):
    if x['pct'] > 0.5:
        return 1
    return 0


def compute_profit(df,y_pred,y_true,bet, y_type, threshold=0.5):
    fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total = simple_betting_profits(df.copy(), bet=bet)
    if y_type == 'single':
        combined = pd.DataFrame({'pred': y_pred, 'act': y_true, 'winning_odds': df['winning_odds']})
        combined['bet'] = -bet
        # print('len', len(combined))
        print('pred:',combined.pred.sum())
        # print('act:',combined.act.sum())
        combined['profit'] = combined.apply(lambda x: correct_single(x) , axis=1)
        # print('profit:', combined.profit.sum())
        btb_profit_total = combined['profit'].sum()
        combined = combined.sort_values(by='profit', ascending=False)
        # print(combined.head(50))
        # print(combined.tail(50))
        return btb_profit_total, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total
    if y_type == 'multi':
        # code for converting 3 to 1
        # y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
        # variance = abs(y_pred - y_true)
        # diff = np.sum(a=variance, axis=1)
        # combined = pd.DataFrame({'diff': diff, 'winning_odds': df['winning_odds']})
        # combined['bet'] = -bet
        # combined['profit'] = combined.apply(lambda x: correct_multi(x) , axis=1)
        # btb_profit_total = combined['profit'].sum()
        # return btb_profit_total, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total
        combined = pd.DataFrame({'pct': y_pred, 'act': y_true, 'winning_odds': df['winning_odds']})
        combined['bet'] = -bet
        combined['threshold'] = threshold
        combined['profit'] = combined.apply(lambda x: correct_threshold(x), axis=1)
        combined['predict'] = combined.apply(lambda x: round_to(x), axis=1)
        btb_profit_total = combined['profit'].sum()
        scores = compute_overall_scores(y_true, combined.predict)
        print(btb_profit_total)
        return scores, btb_profit_total, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total









