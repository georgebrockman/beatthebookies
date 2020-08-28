






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
