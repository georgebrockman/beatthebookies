import itertools
import os
import pandas as pd
import numpy as np
import sqlite3

from beatthebookies.utils import simple_time_tracker



def get_rankings(df):

    root_dir = os.path.dirname(os.path.dirname(__file__))
    sql_path = os.path.join(root_dir, 'beatthebookies/', 'data/')
    # sql_path = '../data/'
    database = sql_path + 'database.sqlite'
    conn = sqlite3.connect(database)

    home_stats = pd.read_sql(""" SELECT ta.date, ta.team_api_id, ta.buildUpPlaySpeed as bups,
                          buildUpPlayPassing as bupp, chanceCreationPassing as ccp, chanceCreationCrossing as ccc,
                          chanceCreationShooting as ccs, defencePressure as dp, defenceAggression as da, defenceTeamWidth as dtw
                          FROM Team_Attributes as ta""", conn)

    home_stats['date'] = pd.to_datetime(home_stats['date'])
    home_stats['year'] = pd.DatetimeIndex(home_stats['date']).year
    season_maps = {2010: '2009/2010', 2011: '2010/2011', 2012:'2011/2012',
                 2013:'2013/2014', 2014: '2014/2015', 2015:'2015/2016'}
    home_stats['season'] = home_stats['year'].map(season_maps)
    home_stats = home_stats.drop(columns=['date','year'])
    away_stats = home_stats.copy()

    df = pd.merge(df, home_stats, how='left', left_on=['season', 'home_api'], right_on=['season', 'team_api_id'])
    df = df.drop(columns=['home_api','team_api_id'])
    df = pd.merge(df, away_stats, how='left', left_on=['season', 'away_api'], right_on=['season', 'team_api_id'])
    df = df.drop(columns=['away_api','team_api_id'])

    return df



def get_opp_totals(df):

    def opposite_stat(x, team='away_team', stat='home_team_goal', add='away_t_away_goals'):
        if x['stage'] == 1:
            return 0
        opp_team = 'away_team' if team == 'home_team' else 'home_team'
        opp_stat = df[(df['season'] == x['season']) & (df['date'] < x['date']) & (df[opp_team] == x[team])]\
            .groupby(opp_team)[stat].sum() + x[add]
        if len(opp_stat) > 0:
            return opp_stat[0]
        return 0


    assign_zero = ['home_t_total_goals', 'home_t_total_goals_against', 'home_t_total_shots', 'home_t_total_shots_against',
                    'home_t_total_shots_ot', 'home_t_total_shots_ot_against', 'home_t_total_fouls', 'home_t_total_fouls_against',
                    'home_t_total_corn','home_t_total_corn_against', 'home_t_total_yel', 'home_t_total_yel_against', 'home_t_total_red',
                    'home_t_total_red_against', 'home_t_total_wins', 'home_t_total_losses','home_t_total_draws','away_t_total_goals',
                    'away_t_total_goals_against', 'away_t_total_shots', 'away_t_total_shots_against', 'away_t_total_shots_ot',
                    'away_t_total_shots_ot_against', 'away_t_total_fouls', 'away_t_total_fouls_against','away_t_total_corn',
                    'away_t_total_corn_against', 'away_t_total_yel', 'away_t_total_yel_against', 'away_t_total_red','away_t_total_red_against',
                     'away_t_total_wins', 'away_t_total_losses','away_t_total_draws']

    for col in assign_zero:
        df[col] = 0

    # home team opposite stats
    df['home_t_total_goals'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='away_team_goal', add='home_t_home_goals' ), axis=1)
    df['home_t_total_goals_against'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='home_team_goal', add='home_t_home_goals_against' ), axis=1)
    df['home_t_total_shots' ] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='away_shots', add='home_t_home_shots'), axis=1)
    df['home_t_total_shots_against'] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='home_shots', add='home_t_home_shots_against'), axis=1)
    df['home_t_total_shots_ot' ] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='away_shots_ot', add='home_t_home_shots_ot'), axis=1)
    df['home_t_total_shots_ot_against' ] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='home_shots_ot', add='home_t_home_shots_ot_against'), axis=1)
    df['home_t_total_fouls' ] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='away_fouls', add='home_t_home_fouls'), axis=1)
    df['home_t_total_fouls_against'] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='home_fouls', add='home_t_home_fouls_against'), axis=1)
    df['home_t_total_corn'] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='away_corn', add='home_t_home_corn'), axis=1)
    df['home_t_total_corn_against' ] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='home_corn', add='home_t_home_corn_against'), axis=1)
    df['home_t_total_yel' ] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='away_yel', add='home_t_home_yel'), axis=1)
    df['home_t_total_yel_against' ] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='home_yel', add='home_t_home_yel_against'), axis=1)
    df['home_t_total_red'] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='away_red', add='home_t_home_red'), axis=1)
    df['home_t_total_red_against'] = df.apply(lambda x: opposite_stat(x, team='home_team', stat='home_red', add='home_t_home_red_against'), axis=1)

    df['home_t_total_wins'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='away_w', add='home_t_home_wins' ), axis=1)
    df['home_t_total_losses'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='home_w', add='home_t_home_losses' ), axis=1)
    df['home_t_total_draws'] = df.apply(lambda x: opposite_stat( x, team='home_team', stat='draw', add='home_t_home_draws'), axis=1)
    # # away team opposite stats
    df['away_t_total_goals'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='home_team_goal', add='away_t_away_goals' ), axis=1)
    df['away_t_total_goals_against'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='away_team_goal', add='away_t_away_goals_against' ), axis=1)
    df['away_t_total_shots' ] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='home_shots', add='away_t_away_shots'), axis=1)
    df['away_t_total_shots_against'] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='away_shots', add='away_t_away_shots_against'), axis=1)
    df['away_t_total_shots_ot' ] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='home_shots_ot', add='away_t_away_shots_ot'), axis=1)
    df['away_t_total_shots_ot_against' ] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='away_shots_ot', add='away_t_away_shots_ot_against'), axis=1)
    df['away_t_total_fouls' ] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='home_fouls', add='away_t_away_fouls'), axis=1)
    df['away_t_total_fouls_against'] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='away_fouls', add='away_t_away_fouls_against'), axis=1)
    df['away_t_total_corn'] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='home_corn', add='away_t_away_corn'), axis=1)
    df['away_t_total_corn_against' ] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='away_corn', add='away_t_away_corn_against'), axis=1)
    df['away_t_total_yel' ] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='home_yel', add='away_t_away_yel'), axis=1)
    df['away_t_total_yel_against' ] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='away_yel', add='away_t_away_yel_against'), axis=1)
    df['away_t_total_red'] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='home_red', add='away_t_away_red'), axis=1)
    df['away_t_total_red_against'] = df.apply(lambda x: opposite_stat(x, team='away_team', stat='away_red', add='away_t_away_red_against'), axis=1)

    df['away_t_total_wins'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='home_w', add='away_t_away_wins'), axis=1)
    df['away_t_total_losses'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='away_w', add='away_t_away_losses'), axis=1)
    df['away_t_total_draws'] = df.apply(lambda x: opposite_stat( x, team='away_team', stat='draw', add='away_t_away_draws'), axis=1)

    return df

def get_loc_totals(df):
    # home team goals
    df['home_t_home_goals'] = df.groupby(['season','home_team'])['home_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_goals_against'] = df.groupby(['season','home_team'])['away_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # # home team shots
    df['home_t_home_shots'] = df.groupby(['season','home_team'])['home_shots'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_shots_ot'] = df.groupby(['season','home_team'])['home_shots_ot'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_shots_against'] = df.groupby(['season','home_team'])['away_shots'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_shots_ot_against'] = df.groupby(['season','home_team'])['away_shots_ot'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # # home fouls
    df['home_t_home_fouls'] = df.groupby(['season','home_team'])['home_fouls'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_fouls_against'] = df.groupby(['season','home_team'])['away_fouls'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_corn'] = df.groupby(['season','home_team'])['home_corn'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_corn_against'] = df.groupby(['season','home_team'])['away_corn'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_yel'] = df.groupby(['season','home_team'])['home_yel'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_yel_against'] = df.groupby(['season','home_team'])['away_yel'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_red'] = df.groupby(['season','home_team'])['home_red'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_red_against'] = df.groupby(['season','home_team'])['away_red'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # # home team win stats
    df['home_t_home_wins'] = df.groupby(['season','home_team'])['home_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_losses'] = df.groupby(['season','home_team'])['away_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['home_t_home_draws'] = df.groupby(['season','home_team'])['draw'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # away team goal stats
    df['away_t_away_goals'] = df.groupby(['season','away_team'])['away_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_goals_against'] = df.groupby(['season','away_team'])['home_team_goal'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # # away team shots
    df['away_t_away_shots'] = df.groupby(['season','away_team'])['away_shots'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_shots_ot'] = df.groupby(['season','away_team'])['away_shots_ot'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_shots_against'] = df.groupby(['season','away_team'])['home_shots'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_shots_ot_against'] = df.groupby(['season','away_team'])['home_shots_ot'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # # away fouls
    df['away_t_away_fouls'] = df.groupby(['season','away_team'])['away_fouls'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_fouls_against'] = df.groupby(['season','away_team'])['home_fouls'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_corn'] = df.groupby(['season','away_team'])['away_corn'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_corn_against'] = df.groupby(['season','away_team'])['home_corn'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_yel'] = df.groupby(['season','away_team'])['away_yel'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_yel_against'] = df.groupby(['season','away_team'])['home_yel'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_red'] = df.groupby(['season','away_team'])['away_red'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_red_against'] = df.groupby(['season','away_team'])['home_red'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    # # away team win stats
    df['away_t_away_wins'] = df.groupby(['season','away_team'])['away_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_losses'] = df.groupby(['season','away_team'])['home_w'].apply(lambda x  : x.cumsum().shift(fill_value=0))
    df['away_t_away_draws'] = df.groupby(['season','away_team'])['draw'].apply(lambda x  : x.cumsum().shift(fill_value=0))


    return df


def get_winner(df):
    df['home_w'] = 0
    df['draw'] = 0
    df['away_w'] = 0
    # set winner sql
    # df.loc[df['home_team_goal'] > df['away_team_goal'], 'home_w'] = 1
    # df.loc[df['home_team_goal'] < df['away_team_goal'], 'away_w'] = 1
    # df.loc[df['home_team_goal'] == df['away_team_goal'], 'draw'] = 1

    # set winner csv
    df.loc[df['FTR'] == 'H', 'home_w'] = 1
    df.loc[df['FTR'] == 'D', 'draw'] = 1
    df.loc[df['FTR'] == 'A' , 'away_w'] = 1

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
                                m.away_team_goal, m.home_team_api_id as home_api, m.away_team_api_id as away_api, m.WHH, m.WHD, m.WHA
                                FROM Match as m
                                LEFT JOIN Team AS ht on ht.team_api_id = m.home_team_api_id
                                LEFT JOIN Team AS at on at.team_api_id = m.away_team_api_id
                                WHERE league_id = :league AND season != :season
                                ;""", conn, params={"league":league, "season":season})
    else:
        df = pd.read_sql("""SELECT m.id,
                                m.season, m.stage, m.date,
                                ht.team_long_name as home_team, at.team_long_name as away_team, m.home_team_goal,
                                m.away_team_goal, m.home_team_api_id as home_api, m.away_team_api_id as away_api, m.WHH, m.WHD, m.WHA
                                FROM Match as m
                                LEFT JOIN Team AS ht on ht.team_api_id = m.home_team_api_id
                                LEFT JOIN Team AS at on at.team_api_id = m.away_team_api_id
                                WHERE league_id = :league AND season = :season
                                ;""", conn, params={"league":league, "season":season})

    df.sort_values('date', inplace=True)
    # df = get_winner(df)
    # df = get_loc_totals(df)
    # df = get_opp_totals(df)
    # df = get_rankings(df)

    return df









@simple_time_tracker
def get_csv_data(season='2015/2016', league=1729, full=False, local=False, optimize=False, **kwargs):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(root_dir, 'beatthebookies/', 'data/')
    file = csv_path + 'newdata.csv'
    df = pd.read_csv(file)
    df.drop(columns='Unnamed: 0', inplace=True)
    df.rename(columns={'Date': 'date', 'Season': 'season','HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'FTHG': 'home_team_goal', 'FTAG': 'away_team_goal',
                        'HS': 'home_shots', 'HST': 'home_shots_ot', 'HC': 'home_corn', 'AC': 'away_corn', 'AS': 'away_shots', 'AST': 'away_shots_ot',
                        'HF': 'home_fouls', 'AF': 'away_fouls', 'HY': 'home_yel', 'AY': 'away_yel', 'HR': 'home_red', 'AR': 'away_red'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # number of weeks in the season
    lst = range(1,39)
    # number of games in a week
    season = list(itertools.chain.from_iterable(itertools.repeat(x, 10) for x in lst))
    # of seasons in dataset
    full = list(itertools.chain.from_iterable(itertools.repeat(season, 12)))
    df['stage'] = full

    df = get_winner(df)
    df = get_loc_totals(df)
    df = get_opp_totals(df)

    return df

def get_prem_league():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(root_dir, 'beatthebookies/', 'data/')
    file = csv_path + 'premiertotals.csv'
    df = pd.read_csv(file)

    return df


@simple_time_tracker
def get_betting_data(season='2015/2016', league=1729, local=False, optimize=False, **kwargs):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    sql_path = os.path.join(root_dir, 'beatthebookies/', 'data/')
    database = sql_path + 'database.sqlite'
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








