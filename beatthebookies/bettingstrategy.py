import pandas as pd
import numpy as np


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

def correct_kelly(x):
    if x['act'] == 1:
        return abs(x['bet'] * x['winning_odds']) - x['bet']
    return - x['bet']

def compute_stake(x):
    stake_pct = ((x['B'] * x['prob_succ']) -(1 - x['prob_succ'])) / x['B']
    if stake_pct <= 0:
        return 0
    return stake_pct

def kelly_prediction(y_pred, odds, bet):
    odds = odds
    prob_succ = y_pred
    B = odds = 1
    stake_pct = ((B * y_pred) - (1 - prob_succ)) / B
    stake = round(stake_pct * bet,2)
    if stake < 0:
        return 0
    return stake


def optimizeddog(df, y_pred, y_true, bankroll=100):
    """ This function determines the optimum split of your betting bankroll for each Premier League Game Week """
    under_odd = df[['WHH','WHD','WHA']].max()
    B = under_odd - 1
    combined_dog = pd.DataFrame({'odds': odds, 'B': B, 'prob_succ': y_pred})
    combined_dog['stake_pct'] = combined_dog.apply(lambda x: compute_stake(x), axis=1)
    combined_Dog['stake'] = combined_dog.stake_pct * bankroll
    return combined_dog.stake

def optimizeddogprofit(df, y_pred, y_true, bankroll=10):
    stake = optimizeddog(df, y_pred, y_true, bankroll=bankroll)
    # create the combined data frame
    combined_dog_p = pd.DataFrame({'pred': y_pred, 'act': y_true, 'winning_odds': df[['WHH','WHD','WHA']].max(), 'bet': stake})
    combined_dog_p['profit'] = combined_dog_p.apply(lambda x: correct_kelly(x) , axis=1)
    kdog_profit = combined_dog_p.profit.sum()
    print("kelly dog", kdog_profit)
    return kdog_profit # rename


def optimizedhomebet(df, y_pred, y_true, bankroll=10): # bankroll for the game week
    """ This function determines the optimum split of your betting bankroll for each Premier League Game Week """
    odds = df['WHH']
    B = odds - 1
    # prob_succ = y_pred
    combined = pd.DataFrame({'odds': odds, 'B': B, 'prob_succ': y_pred})
    combined['stake_pct'] = combined.apply(lambda x: compute_stake(x), axis=1)
    combined['stake'] = combined.stake_pct * bankroll
    return combined.stake

def optimizedhomeprofit(df, y_pred, y_true, bankroll=10):
    stake = optimizedhomebet(df, y_pred, y_true, bankroll=bankroll)
    # create the combined data frame
    combined_p = pd.DataFrame({'pred': y_pred, 'act': y_true, 'winning_odds': df['WHH'], 'bet': stake})
    combined_p['profit'] = combined_p.apply(lambda x: correct_kelly(x) , axis=1)
    k_prof = combined_p.profit.sum()
    print("kelly home", k_prof )
    return k_prof  # rename

def compute_profit(df,y_pred,y_true,bet):
    fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total = simple_betting_profits(df.copy(), bet=bet)
    combined = pd.DataFrame({'pred': y_pred, 'act': y_true, 'winning_odds': df['winning_odds']})
    combined['bet'] = -bet
    combined['profit'] = combined.apply(lambda x: correct_bet(x) , axis=1)
    btb_profit_total = combined['profit'].sum()
    combined = combined.sort_values(by='profit', ascending=False)
    print("btb", btb_profit_total)
    return btb_profit_total, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total









