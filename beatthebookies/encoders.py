import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# class CustomStandardScaler(BaseEstimator, TransformerMixin):

#     def __init__(self):
#         self.mean = None
#         self.std = None

#     def fit(self, X):
#         self.mean = np.nanmean(X, axis=0)
#         self.std = np.nanstd(X, axis=0)

#     def transform(self, X):
#         return (X - self.mean) / self.std

# class CustomNormaliser(BaseEstimator, TransformerMixin):

#     def __init__(self):
#         self.min = None
#         self.max = None

#     def fit(self, X):
#         self.min = np.nanmin(X, axis=0)
#         self.max = np.nanmax(X, axis=0)

#     def transform(self, X):
#         return (X - self.min) / (self.max - self.min)

class FifaDifferentials(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.H_ATT = "H_ATT"
        self.H_MID = "H_MID"
        self.H_DEF = "H_DEF"
        self.H_OVR = "H_OVR"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # home team differentials
        X['H_ATT_D'] = X['H_ATT'] - X['A_ATT']
        X['H_MID_D'] = X['H_MID'] - X['A_MID']
        X['H_DEF_D'] = X['H_DEF'] - X['A_DEF']
        X['H_OVR_D'] = X['H_OVR'] - X['A_OVR']

        X['H_G_D'] = X['home_t_total_goals'] - X['away_t_total_goals']
        X['H_GA_D'] = X['home_t_total_goals_against'] - X['away_t_total_goals_against']
        X['H_S_D'] = X['home_t_total_shots'] - X['away_t_total_shots']
        X['H_SA_D'] = X['home_t_total_shots_against'] - X['away_t_total_shots_against']

        return X[['H_ATT_D', 'H_MID_D', 'H_DEF_D', 'H_OVR_D', 'H_G_D', 'H_GA_D', 'H_S_D', 'H_SA_D']]

# class FifaDifferentials2(BaseEstimator, TransformerMixin):

#     def __init__(self, df):
#         self.df = df

#     def fit(self, df, y=None):
#         return self

#     def transform(self, df, y=None):
#         assert isinstance(df, pd.DataFrame)
#         # home team differentials
#         df['H_ATT_D'] = df['H_ATT'] - df['A_ATT']
#         df['H_MID_D'] = df['H_MID'] - df['A_MID']
#         df['H_DEF_D'] = df['H_DEF'] - df['A_DEF']
#         df['H_OVR_D'] = df['H_OVR'] - df['A_OVR']

#         return df

# class GoalDifferentials(BaseEstimator, TransformerMixin):

#     def __init__(self, df):
#         self.df = df

#     def fit(self, df, y=None):
#         return self

#     def transform(self, df, y=None):
#         assert isinstance(selfm df, y=None)


class WeeklyGoalAverages(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.home_t_total_goals = 'home_t_total_goals'
        self.away_t_total_goals = 'away_t_total_goals'
        self.prev_home_matches = 'prev_home_matches'
        self.prev_away_matches = 'prev_away_matches'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['average_home_t_goals'] = X['home_t_total_goals'] / X['prev_home_matches']
        X['average_away_t_goals'] = X['away_t_total_goals'] / X['prev_away_matches']
        # dividing by zero returns infinite and NaN values
        X.replace([np.inf, np.nan], 0, inplace=True)

        return X[['average_home_t_goals','average_away_t_goals']]









