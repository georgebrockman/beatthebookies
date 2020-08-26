# import pandas as pd
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin

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
