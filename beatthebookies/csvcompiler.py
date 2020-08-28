






# not accurate code yet

# create a dictionary to map need all value pairs
# team_dict = {'West Bromwich': 'West Brom', 'West Bromwich Albion': 'West Brom',
#              'Arsenal FC': 'Arsenal', 'Chelsea FC':'Chelsea', 'Chelsea':'Chelsea',
#              'Reading FC': 'Reading', 'Reading':'Reading', 'Arsenal':'Arsenal', 'Bournemouth':'Bournemouth',
#              'AFC Bournemouth': 'Bournemouth', 'Sheffield United': 'Sheffield United',
#              'Manchester United': 'Man United', 'Liverpool' : 'Liverpool', 'Newcastle United': 'Newcastle',
#              'Middlesbrough':'Middlesbrough', 'Bolton Wanderers':'Bolton', 'Everton':'Everton',
#              'Hull City': 'Hull', 'Sunderland': 'Sunderland', 'West Ham United': 'West Ham',
#              'Aston Villa':'Aston Villa', 'Blackburn Rovers':'Blackburn', 'Fulham':'Fulham',
#              'Stoke City':'Stoke', 'Tottenham Hotspur': 'Tottenham', 'Manchester City': 'Man City',
#              'Wigan Athletic': 'Wigan', 'Portsmouth':'Portsmouth', 'Wolverhampton Wanderers':'Wolves',
#              'Birmingham City': 'Birmingham', 'Burnley': 'Burnley', 'Blackpool':'Blackpool',
#              'Queens Park Rangers': 'QPR', 'Swansea City':'Swansea', 'Norwich City': 'Norwich', 'Southampton':'Southampton',
#              'Crystal Palace':'Crystal Palace', 'Cardiff City': 'Cardiff', 'Leicester City':'Leicester',
#              'Watford':'Watford', 'Brighton & Hove Albion': 'Brighton', 'Huddersfield Town': 'Huddersfield'
#             }

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
