import itertools
import os
import pandas as pd
import numpy as np
import sqlite3

from beatthebookies.utils import simple_time_tracker


# not accurate code yet
    # def csv_compile(*args):
    #     """ function takes in DataFrames, specifies the desired columns and concatonates into one DataFrame """
    #     # columns for cvs file
    #     columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
    #        'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HF',
    #        'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'WHH', 'WHD', 'WHA']
    #     for x in *args:
    #         return x = df[columns]
    #     # concatonate the dataframes
    #     df = pd.concat([x])
    #     df = reset_index(drop=True, inplace=True)
          # remove any rows with NaN values

    #     return df

    # def csv_add_season(df):
    #     """ function updates a column to add season and stage (Game Week) """
    #     pass


    # def csv_merg(df, df1):
    #     """ Function adds the home team and away team attributes for each match where df= match data and df1 = fifa ratings"""

    #     df = pd.merge(df, df1, how='left', left_on=['Season', 'HomeTeam'], right_on=['Season', 'Team'])
    #     # drop league column, unamed and team
    #     df = df.drop(columns=['League', 'Unnamed: 0_y', 'Unnamed: 0_x', 'Team'])
    #     # change name of these new columns to home team
    #     df.rename(columns={"ATT": "H_ATT", "MID":"H_MID","DEF":"H_DEF", "OVR":"H_OVR"}, inplace=True)
    #     # add away team stats
    #     df = pd.merge(df, ratings, how='left', left_on=['Season', 'AwayTeam'], right_on=['Season', 'Team'])
    #     # drop columns
    #     df = df.drop(columns=['League', 'Unnamed: 0', 'Team'])
    #     # change name of these new columns to home team
    #     df.rename(columns={"ATT": "A_ATT", "MID":"A_MID","DEF":"A_DEF", "OVR":"A_OVR"}, inplace=True)

    #     return df


def get_opp_totals(df):
    print('getting opposite totals')

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
    print('getting location totals')
    def get_week(x):
        opp_stat = df[(df['season'] == x['season']) & (df['date'] < x['date']) & (df['away_team'] == x['home_team'])]\
            .groupby('away_team')['away_team'].count() + x['home_t_prev_home_matches'] + 1
        if len(opp_stat) > 0:
            return opp_stat[0]
        return 1



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
    #
    df['home_t_prev_home_matches'] = df.groupby(['season','home_team']).cumcount()
    df['away_t_prev_away_matches'] = df.groupby(['season','away_team']).cumcount()
    df['stage'] = df.apply(lambda x: get_week(x), axis=1)

    return df

def get_winner(df):
    df['home_w'] = 0
    df['draw'] = 0
    df['away_w'] = 0
    # set winner
    df.loc[df['FTR'] == 'H', 'home_w'] = 1
    df.loc[df['FTR'] == 'D', 'draw'] = 1
    df.loc[df['FTR'] == 'A' , 'away_w'] = 1

    return df



@simple_time_tracker
def get_csv_data():
    print('loading csv')
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
    df = get_winner(df)
    df = get_loc_totals(df)
    df = get_opp_totals(df)

    return df


if __name__ == "__main__":
    df = get_csv_data()
    df.to_csv("beatthebookies/data/premiertotals.csv", index=False, encoding='utf-8-sig')




