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

    def __init__(self, H_ATT, H_MID. H_DEF, H_OVR):
        self.H_ATT = H_ATT
        self.H_MID = H_MID
        self.H_DEF = H_DEF
        self.H_OVR = H_OVR

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None)
        assert isinstance(X, pd.DataFrame)
        # home team differentials
        X['H_ATT_D'] = X['H_ATT'] - X['A_ATT']
        X['H_MID_D'] = X['H_MID'] - X['A_MID']
        X['H_DEF_D'] = X['H_DEF'] - X['A_DEF']
        X['H_OVR_D'] = X['H_OVR'] - X['A_OVR']

        return X

class FifaDifferentials2(BaseEstimator, TransformerMixin):

    def __init__(self, df):
        self.df = df

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None)
        assert isinstance(df, pd.DataFrame)
        # home team differentials
        df['H_ATT_D'] = df['H_ATT'] - df['A_ATT']
        df['H_MID_D'] = df['H_MID'] - df['A_MID']
        df['H_DEF_D'] = df['H_DEF'] - df['A_DEF']
        df['H_OVR_D'] = df['H_OVR'] - df['A_OVR']

        return df







