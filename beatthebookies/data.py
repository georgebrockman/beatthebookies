import os
import pandas as pd
import numpy as np
import sqlite3

from beatthebookies.utils import simple_time_tracker


def get_opp_totals(df):

    def opposite_stat(x, team='away_team', stat='home_team_goal'):
        if x['stage'] == 1:
            return 0
        opp_team = 'away_team' if team == 'home_team' else 'home_team'
        opp_stat = df[(df['stage'] < x['stage']) & (df[opp_team] == x[team])]\
            .groupby(['season', opp_team])[stat].sum()
        if len(opp_stat) > 0:
            return opp_stat[0]
        return 0

    assign_zero = ['home_t_away_goals', 'home_t_away_goals_against','home_t_away_wins', 'home_t_away_losses',
                  'home_t_away_draws','away_t_home_goals','away_t_home_goals_against', 'away_t_home_wins', 'away_t_home_losses',
                  'away_t_home_draws']

    for col in assign_zero:
        df[col] = 0

    # home team opposite stats
    df['home_t_away_goals'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='away_team_goal' ), axis=1)
    df['home_t_away_goals_against'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='home_team_goal' ), axis=1)
    df['home_t_away_wins'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='away_w' ), axis=1)
    df['home_t_away_losses'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='home_w' ), axis=1)
    df['home_t_away_draws'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='draw'), axis=1)
    # away team opposite stats
    df['away_t_home_goals'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='home_team_goal' ), axis=1)
    df['away_t_home_goals_against'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='away_team_goal' ), axis=1)
    df['away_t_home_wins'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='home_w' ), axis=1)
    df['away_t_home_losses'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='away_w'), axis=1)
    df['away_t_home_draws'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='draw' ), axis=1)

    return df

def get_loc_totals(df):
    # home team goal stats
    df['home_t_home_goals'] = df.groupby(['season','home_team'])['home_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_goals_against'] = df.groupby(['season','home_team'])['away_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # home team win stats
    df['home_t_home_wins'] = df.groupby(['season','home_team'])['home_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_losses'] = df.groupby(['season','home_team'])['away_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_draws'] = df.groupby(['season','home_team'])['draw'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # away team goal stats
    df['away_t_away_goals'] = df.groupby(['season','away_team'])['away_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_goals_against'] = df.groupby(['season','away_team'])['home_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # away team win stats
    df['away_t_away_wins'] = df.groupby(['season','away_team'])['away_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_losses'] = df.groupby(['season','away_team'])['home_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_draws'] = df.groupby(['season','away_team'])['draw'].apply(lambda x  : x.cumsum().shift(fill_value=0))

    return df


def get_winner(df):
    df['home_w'] = 0
    df['away_w'] = 0
    df['draw'] = 0
    # set winner
    df.loc[df['home_team_goal'] > df['away_team_goal'], 'home_w'] = 1
    df.loc[df['home_team_goal'] < df['away_team_goal'], 'away_w'] = 1
    df.loc[df['home_team_goal'] == df['away_team_goal'], 'draw'] = 1

    return df


@simple_time_tracker
def get_data(season='2015/2016', league=1729, full=False, local=False, optimize=False, **kwargs):
    ''' takes season to ignore and specific league '''
    root_dir = os.path.dirname(os.path.dirname(__file__))
    sql_path = os.path.join(root_dir, 'beatthebookies/', 'data/')
    # sql_path = '../data/'
    database = sql_path + 'database.sqlite'
    conn = sqlite3.connect(database)
    if full:
        df = pd.read_sql("""SELECT m.id,
                                m.season, m.stage, m.date,
                                ht.team_long_name as home_team, at.team_long_name as away_team, m.home_team_goal,
                                m.away_team_goal
                                FROM Match as m
                                LEFT JOIN Team AS ht on ht.team_api_id = m.home_team_api_id
                                LEFT JOIN Team AS at on at.team_api_id = m.away_team_api_id
                                WHERE league_id = :league AND season != :season
                                ;""", conn, params={"league":league, "season":season})
    else:
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
    df = get_winner(df)
    df = get_loc_totals(df)
    df = get_opp_totals(df)

    return df

@simple_time_tracker
def get_betting_data(season='2015/2016', league=1729, local=False, optimize=False, **kwargs):
    path = "../data/"
    database = path + 'database.sqlite'
    conn = sqlite3.connect(database)

    df = pd.read_sql("""SELECT m.id,
                            m.season, m.stage, m.date,
                            ht.team_long_name as home_team, at.team_long_name as away_team, m.home_team_goal,
                            m.away_team_goal, m.WHH, m.WHD, m.WHA
                            FROM Match as m
                            LEFT JOIN Team AS ht on ht.team_api_id = m.home_team_api_id
                            LEFT JOIN Team AS at on at.team_api_id = m.away_team_api_id
                            WHERE league_id = :league AND season = :season
                            ;""", conn, params={"league":league, "season":season})
    # add win columns
    df = get_winner(df)
    # sort into order
    df.sort_values('date', inplace=True)

    return df




if __name__ == "__main__":
    df = get_data(season='2008/2009')
    # dfb = get_betting_data()








