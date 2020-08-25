import pandas as pd
import numpy as np
import sqlite3

from beatthebookies.utils import simple_time_tracker


def get_full_totals(df):
    def ht_total_goals(st, ht, hg):
        ''' finds away goals scored by home team '''
        if st == 1:
            return 0
        total_goals = df[(df['stage'] < st) & (df['away_team'] == ht)]\
            .groupby('away_team')['away_team_goal'].sum() + hg
        if len(total_goals) > 0:
            return total_goals[0]
        return 0

    def ht_total_goals_agst(st, ht, hga):
        ''' finds away goals scored against the home team '''
        if st == 1:
            return 0
        total_goals = df[(df['stage'] < st) & (df['away_team'] == ht)]\
            .groupby('away_team')['home_team_goal'].sum() + hga
        if len(total_goals) > 0:
            return total_goals[0]
        return 0

    def ht_total_wins(st, ht, hw):
        ''' finds away wins for the home team '''
        if st == 1:
            return 0
        total_wins = df[(df['stage'] < st) & (df['away_team'] == ht)]\
            .groupby('away_team')['away_w'].sum() + hw
        if len(total_wins) >0:
            return total_wins[0]
        return 0

    def ht_total_losses(st, ht, hl):
        ''' finds away losses for the home team '''
        if st == 1:
            return 0
        total_losses = df[(df['stage'] < st) & (df['away_team'] == ht)]\
            .groupby('away_team')['home_w'].sum() + hl
        if len(total_losses) > 0:
            return total_losses[0]
        return 0

    def ht_total_draws(st, ht, hd):
        ''' finds away draws for the home team '''
        if st == 1:
            return 0
        total_draws = df[(df['stage'] < st) & (df['away_team'] == ht)]\
            .groupby('away_team')['draw'].sum() + hd
        if len(total_draws) > 0:
            return total_draws[0]
        return 0


    def at_total_goals(st, at, ag):
        ''' finds home goals scored by away team '''
        if st == 1:
            return 0
        total_goals = df[(df['stage'] < st) & (df['home_team'] == at)]\
            .groupby('home_team')['home_team_goal'].sum() + ag
        if len(total_goals) > 0:
            return total_goals[0]
        return 0

    def at_total_goals_agst(st, at, aga):
        ''' finds home goals scored against the away team '''
        if st == 1:
            return 0
        total_goals = df[(df['stage'] < st) & (df['home_team'] == at)]\
            .groupby('home_team')['away_team_goal'].sum() + aga
        if len(total_goals) > 0:
            return total_goals[0]
        return 0

    def at_total_wins(st, at, aw):
        ''' finds home wins for the away team '''
        if st == 1:
            return 0
        total_wins = df[(df['stage'] < st) & (df['home_team'] == at)]\
            .groupby('home_team')['home_w'].sum() + aw
        if len(total_wins) >0:
            return total_wins[0]
        return 0

    def at_total_losses(st, at, al):
        ''' finds home losses for the away team '''
        if st == 1:
            return 0
        total_losses = df[(df['stage'] < st) & (df['home_team'] == at)]\
            .groupby('home_team')['away_w'].sum() + al
        if len(total_losses) > 0:
            return total_losses[0]
        return 0

    def at_total_draws(st, at, ad):
        ''' finds home draws for the away team '''
        if st == 1:
            return 0
        total_draws = df[(df['stage'] < st) & (df['home_team'] == at)]\
            .groupby('home_team')['draw'].sum() + ad
        if len(total_draws) > 0:
            return total_draws[0]
        return 0


    df['home_t_total_goals'] = df.apply(lambda x: ht_total_goals( x['stage'], x['home_team'], x['home_t_home_goals']), axis=1)
    df['home_t_total_goals_against'] = df.apply(lambda x: ht_total_goals_agst( x['stage'], x['home_team'], x['home_t_home_goals_against']), axis=1)
    df['home_t_total_wins'] = df.apply(lambda x: ht_total_wins( x['stage'], x['home_team'], x['home_t_home_wins']), axis=1)
    df['home_t_total_losses'] = df.apply(lambda x: ht_total_losses( x['stage'], x['home_team'], x['home_t_home_losses']), axis=1)
    df['home_t_total_draws'] = df.apply(lambda x: ht_total_draws( x['stage'], x['home_team'], x['home_t_home_draws']), axis=1)

    df['away_t_total_goals'] = df.apply(lambda x: at_total_goals( x['stage'], x['away_team'], x['away_t_away_goals']), axis=1)
    df['away_t_total_goals_against'] = df.apply(lambda x: at_total_goals_agst( x['stage'], x['away_team'], x['away_t_away_goals_against']), axis=1)
    df['away_t_total_wins'] = df.apply(lambda x: at_total_wins( x['stage'], x['away_team'], x['away_t_away_wins']), axis=1)
    df['away_t_total_losses'] = df.apply(lambda x: at_total_losses( x['stage'], x['away_team'], x['away_t_away_losses']), axis=1)
    df['away_t_total_draws'] = df.apply(lambda x: at_total_draws( x['stage'], x['away_team'], x['away_t_away_draws']), axis=1)

    return df



def get_totals(df):
    df['home_w'] = 0
    df['away_w'] = 0
    df['draw'] = 0
    # set winner
    df.loc[df['home_team_goal'] > df['away_team_goal'], 'home_w'] = 1
    df.loc[df['home_team_goal'] < df['away_team_goal'], 'away_w'] = 1
    df.loc[df['home_team_goal'] == df['away_team_goal'], 'draw'] = 1
    # home team goal stats
    df['home_t_home_goals'] = df.groupby('home_team')['home_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_total_goals'] = 0
    df['home_t_home_goals_against'] = df.groupby('home_team')['away_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_total_goals_against'] = 0
    # home team win stats
    df['home_t_home_wins'] = df.groupby('home_team')['home_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_losses'] = df.groupby('home_team')['away_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_draws'] = df.groupby('home_team')['draw'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_total_wins'] = 0
    df['home_t_total_losses'] = 0
    df['home_t_total_draws'] = 0
    # away team goal stats
    df['away_t_away_goals'] = df.groupby('away_team')['away_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_total_goals'] = 0
    df['away_t_away_goals_against'] = df.groupby('away_team')['home_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_total_goals_against'] = 0
    # away team win stats
    df['away_t_away_wins'] = df.groupby('away_team')['away_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_losses'] = df.groupby('away_team')['home_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_draws'] = df.groupby('away_team')['draw'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_total_wins'] = 0
    df['away_t_total_losses'] = 0
    df['away_t_total_draws'] = 0

    df = get_full_totals(df)

    return df


# @simple_time_tracker
def get_data(season='2015/2016', league=1729, local=False, optimize=False, **kwargs):
    path = "data/"
    database = path + 'database.sqlite'
    conn = sqlite3.connect(database)

    df = pd.read_sql("""SELECT m.id,
                            m.season, m.stage, m.date,
                            ht.team_long_name as home_team, at.team_long_name as away_team, m.home_team_goal,
                            m.away_team_goal
                            FROM Match as m
                            LEFT JOIN Team AS ht on ht.team_api_id = m.home_team_api_id
                            LEFT JOIN Team AS at on at.team_api_id = m.away_team_api_id
                            WHERE league_id = :league AND season = :season
                            ;""", conn, params={"league":league, "season":season})
    df.sort_values('date', inplace=True)

    df = get_totals(df)

    return df




if __name__ == "__main__":
    df = get_data(season='2008/2009')










