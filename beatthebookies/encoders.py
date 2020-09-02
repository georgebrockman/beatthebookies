import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
        X['H_ATT_A'] = X['H_ATT'] - X['A_DEF']
        X['H_DEF_A'] = X['H_DEF'] - X['A_ATT']

        #return X[['H_ATT_D', 'H_MID_D', 'H_DEF_D', 'H_OVR_D', 'H_ATT_A', 'H_DEF_A']]
        return X[['H_MID_D', 'H_OVR_D', 'H_ATT_A', 'H_DEF_A']]


class WeeklyGoalAverages(BaseEstimator, TransformerMixin):

    def __init__(self):
        # self.home_t_total_goals = 'home_t_home_goals'
        # self.away_t_total_goals = 'away_t_away_goals'
        self.home_t_total_goals = 'home_t_total_goals'
        self.away_t_total_goals = 'away_t_total_goals'
        # self.prev_home_matches = 'home_t_prev_home_matches'
        # self.prev_away_matches = 'away_t_prev_away_matches'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # X['home_t_average_home_goals'] = X['home_t_home_goals'] / X['home_t_prev_home_matches']
        # X['away_t_average_away_goals'] = X['away_t_away_goals'] / X['away_t_prev_away_matches']
        X['home_t_average_goals'] = X['home_t_total_goals'] / (X['stage'] - 1)
        X['away_t_average_goals'] = X['away_t_total_goals'] / (X['stage'] - 1)
        # dividing by zero returns infinite and NaN values
        X.replace([np.inf, np.nan], 0, inplace=True)
        X['home_t_average_goals_diff'] = X['home_t_average_goals'] - X['away_t_average_goals']

        # return X[['home_t_average_home_goals', 'home_t_average_goals', 'away_t_average_goals', 'away_t_average_away_goals']]
        return X[['home_t_average_goals_diff']]



class WeeklyGoalAgAverages(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['home_t_average_goals_against'] = X['home_t_total_goals_against'] / (X['stage'] - 1)
        X['away_t_average_goals_against'] = X['away_t_total_goals_against'] / (X['stage'] - 1)
        # dividing by zero returns infinite and NaN values
        X.replace([np.inf, np.nan], 0, inplace=True)
        X['home_t_average_goals_ag_diff'] = X['home_t_average_goals_against'] - X['away_t_average_goals_against']

        return X[['home_t_average_goals_ag_diff']]


class WinPctDifferentials(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['home_t_win_pct'] = X['home_t_total_wins'] / (X['stage'] - 1)
        X['away_t_win_pct'] = X['away_t_total_wins'] / (X['stage'] - 1)
        # dividing by zero returns infinite and NaN values
        X.replace([np.inf, np.nan], 0, inplace=True)
        X['home_t_win_pct_diff'] = X['home_t_win_pct'] - X['away_t_win_pct']

        return X[['home_t_win_pct_diff']]


class HomeAdv(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['home_t_avg_h_g'] = X['home_t_home_goals'] / (X['home_t_prev_home_matches'])
        X['home_t_avg_h_g_a'] = X['home_t_home_goals_against'] / (X['home_t_prev_home_matches'])
        X['away_t_avg_a_g_a'] = X['away_t_away_goals_against'] / (X['away_t_prev_away_matches'])
        X['away_t_avg_a_g'] = X['away_t_away_goals'] / (X['away_t_prev_away_matches'])
        X.replace([np.inf, np.nan], 0, inplace=True)
        X['h_diff_hg'] = X['home_t_avg_h_g'] - X['away_t_avg_a_g_a']
        X['h_diff_hg_a'] = X['home_t_avg_h_g_a'] - X['away_t_avg_a_g']
        # dividing by zero returns infinite and NaN values


        # return X[['home_t_avg_h_g', 'home_t_avg_h_g_a', 'away_t_avg_a_g_a', 'away_t_avg_a_g']]
        return X[['h_diff_hg','h_diff_hg_a']]


class ShotOTPct(BaseEstimator, TransformerMixin):

  def __init__(self):
      pass


  def fit(self, X, y=None):
      return self

  def transform(self, X, y=None):
      assert isinstance(X, pd.DataFrame)
      X['home_t_shototpct'] = X['home_t_total_shots_ot'] / X['home_t_total_shots']
      X['away_t_shototpct'] = X['away_t_total_shots_ot'] / X['away_t_total_shots']
      # dividing by zero returns infinite and NaN values
      X.replace([np.inf, np.nan], 0, inplace=True)
      X['home_t_shototpct_diff'] = X['home_t_shototpct'] - X['away_t_shototpct']

      return X[['home_t_shototpct_diff']]



