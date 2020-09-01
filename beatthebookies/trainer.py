import mlflow
import warnings
import time
import pandas as pd
import numpy as np
from beatthebookies.data import get_data
from beatthebookies.utils import simple_time_tracker, compute_scores, compute_overall_scores
from beatthebookies.encoders import FifaDifferentials, WeeklyGoalAverages, WinPctDifferentials, WeeklyGoalAgAverages, ShotOTPct, HomeAdv
from beatthebookies.bettingstrategy import compute_profit

from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBRegressor, XGBClassifier

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Flatten, SimpleRNN, Conv2D, MaxPooling1D,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from tempfile import mkdtemp
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

from keras.layers import BatchNormalization

# warnings.filterwarnings("ignore", category=FutureWarning)


MLFLOW_URI = "https://mlflow.lewagon.co/"
myname="Chris_Westerman"
EXPERIMENT_NAME = f"[UK][London][{myname}] BeatTheBookies"


class Trainer(object):

    ESTIMATOR = 'logistic'
    rs = RandomUnderSampler(random_state=0)

    def __init__(self, X, y, **kwargs):
        self.pipeline = None
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.X = X
        self.y = y
        self.X_test = self.kwargs.get('X_test', None)
        self.y_test = self.kwargs.get('y_test', None)
        self.y_type = self.kwargs.get('y_type', 'single')
        self.bet = self.kwargs.get('bet', 10)
        # self.le = LabelEncoder()
        # self.le.fit(self.y)
        # self.y = self.le.transform(self.y)
        # self.y_test = self.le.transform(self.y_test)
        # if self.y_type == 'multi':
        #     num_classes = 3
        #     self.y = tensorflow.keras.utils.to_categorical(self.y, num_classes=num_classes)
        #     self.y_test = tensorflow.keras.utils.to_categorical(self.y_test, num_classes=num_classes)

        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=15)
        self.experiment_name = kwargs.get("experiment_name", EXPERIMENT_NAME)
        # self.model_params = None
        self.log_kwargs_params()

        self.X_train, self.y_train = self.balancing(self.X_train, self.y_train)
    def log_kwargs_params(self):
          if self.mlflow:
                self.mlflow_log_param('balance', self.kwargs.get('balance', 'balance'))
                self.mlflow_log_param('gridsearch', self.kwargs.get('gridsearch', False))
                self.mlflow_log_param('bet', self.kwargs.get('bet', 10))




    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        self.mlflow_log_param("model", estimator)
        # added both regressions for predicting scores and classifier for match outcomes
        if estimator == 'Logistic':
            model = LogisticRegression()
        # elif estimator == 'Linear':
        #     model = LinearRegression()
        elif estimator == 'RandomForestClassifier':
            model = RandomForestClassifier()
        # elif estimator == 'RandomForestRegressor':
        #     model = RandomForestRegressor()
        # elif estimator == 'Lasso':
        #     model = Lasso()
        # elif estimator == "Ridge":
        #     model = Ridge()
        elif estimator == "RidgeClassifier":
            model = RidgeClassifier()
        # elif estimator == "GBM":
        #     model = GradientBoostingRegressor()
        elif estimator == "KNNClassifier":
            model = KNeighborsClassifier()
        # elif estimator == "KNNRegressor":
        #     model = KNeighborsRegressor()
        elif estimator == 'GaussianNB':
            model = GaussianNB()
        elif estimator == 'LDA':
            model = LinearDiscriminantAnalysis()
        # elif estimator == "xgboost":
        #     model = XGBRegressor()
        elif estimator == "XGBClassifier":
            model = XGBClassifier()
        elif estimator == "SVC":
            model = SVC(kernel='poly')
        elif estimator == "Sequential":
            model = Sequential()
            model.add(Flatten())
            model.add(BatchNormalization())
            model.add(Dense(32, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='relu',input_shape=(10000,)))
            model.add(Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # model.add(SimpleRNN(1, input_shape=[None, 1], activation='tanh'))
            model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

        else:
            model = LogisticRegression()
        estimator_params = self.kwargs.get("estimator_params", {})
        if estimator != "Sequential":
            model.set_params(**estimator_params)
        return model

    def set_pipeline(self):

        pipe_fifadiff = make_pipeline(FifaDifferentials(), RobustScaler())
        pipe_winpct = make_pipeline(WinPctDifferentials(), StandardScaler())
        pipe_avggoal = make_pipeline(WeeklyGoalAverages(), StandardScaler())
        pipe_shototpct = make_pipeline(ShotOTPct())
        pipe_avggoal_ag = make_pipeline(WeeklyGoalAgAverages(), StandardScaler())
        pipe_home_adv = make_pipeline(HomeAdv(), StandardScaler())


        feateng_blocks = [('fifadiff', pipe_fifadiff, ['H_ATT', 'A_ATT', 'H_MID', 'A_MID', 'H_DEF', 'A_DEF', 'H_OVR', 'A_OVR']),
                          ('windiff', pipe_winpct, ['home_t_total_wins','away_t_total_wins', 'stage']),
                          ('goaldiff', pipe_avggoal, ['home_t_total_goals','away_t_total_goals', 'stage']),
                          ('homeadv', pipe_home_adv, ['home_t_home_goals','home_t_home_goals_against','away_t_away_goals','away_t_away_goals_against',
                                                      'home_t_prev_home_matches', 'away_t_prev_away_matches']),
                          ('shototpct', pipe_shototpct, ['home_t_total_shots', 'home_t_total_shots_ot', 'away_t_total_shots', 'away_t_total_shots_ot']),
                          ('goalagdiff', pipe_avggoal_ag, ['home_t_total_goals_against','away_t_total_goals_against', 'stage'])
                         ]

        features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None, remainder="drop")


        self.pipeline = Pipeline(steps=[
          ('features', features_encoder),
          ('rgs', self.get_estimator())])


    def balancing(self, X_train, y_train):
        ### OVERSAMPLERS
        balance = self.kwargs.get("balance", "SMOTE")

        if balance == "SMOTE":
          # Create new samples without making any disticntion between easy and hard samples to be classified using K-nearest neighbor
          X_train, y_train = SMOTE().fit_resample(X_train, y_train)
          print(Counter(y_train))
          return X_train, y_train
        elif balance == "ADASYN":
          # Create new samples next to the original samples which are wrongly classified by using K-Nearest neighbor
          X_train, y_train = ADASYN().fit_resample(X_train, y_train)
          print(Counter(y_train))
          return X_train, y_train
        elif balance == "RandomOversampler":
          # Duplicating some of the original samples of the minority class
          X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)
          print(Counter(y_train))
          return X_train, y_train

       ### UNDERSAMPLERS
        if balance == "RandomUnderSampler":
          # balances the data by randomly selecting a subset of data for the targeted classes
          X_train, y_train = RandomUnderSampler(random_state=0).fit_resample(X_train, y_train)
          print(Counter(y_train))
          return X_train, y_train

        if balance == "NearMiss":
          # Allows to select 3 different rules of selecting samples based on k-neearest neighbors (version 1,2,3)
          X_train, y_train = NearMiss(version=1).fit_resample(X_train, y_train)
          # print(Counter(y_train))
          return X_train, y_train
        else:
          return X_train, y_train




    @simple_time_tracker
    def train(self):
        tic = time.time()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        self.set_pipeline()
        if self.y_type == 'multi':
            self.pipeline.fit(self.X_train, self.y_train,  rgs__validation_split=0.2, rgs__shuffle=True, rgs__epochs=300,
                        rgs__batch_size=32, rgs__verbose=1, rgs__callbacks=[es])
        else:
            self.pipeline.fit(self.X_train, self.y_train)


    def evaluate(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        bet = self.bet
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_val_pred = self.pipeline.predict(self.X_val)
        # y_test_pred = self.pipeline.predict(self.X_test)
        y_test_pred = self.pipeline.predict(self.X_test).reshape((380,))


        # if self.y_type == 'multi':
        #     idx = np.argmax(y_val_pred, axis=-1)
        #     y_val_pred = np.zeros(y_val_pred.shape)
        #     y_val_pred[np.arange(y_val_pred.shape[0]), idx] = 1

        # scores = compute_overall_scores(y_val_pred, self.y_val)
        # self.mlflow_log_metric("val_precision",scores[1])
        # self.mlflow_log_metric("val_accuracy",scores[0])
        # self.mlflow_log_metric("val_recall",scores[2])
        # self.mlflow_log_metric("val_f1",scores[3])
        if self.y_type == 'single':
            scores = compute_overall_scores(y_test_pred, self.y_test)
            season_profit, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total = compute_profit(self.X_test, y_test_pred, self.y_test, bet, self.y_type, 0.5)

        if self.y_type == 'multi':
            scores, season_profit, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total = compute_profit(self.X_test, y_test_pred, self.y_test, bet, self.y_type, 0.5)

        self.mlflow_log_metric("test_precision",scores[1])
        self.mlflow_log_metric("test_accuracy",scores[0])
        self.mlflow_log_metric("test_recall",scores[2])
        self.mlflow_log_metric("test_f1",scores[3])
        self.mlflow_log_metric("profit_model",season_profit)
        self.mlflow_log_metric("prof_favorites",fav_profit_total)
        self.mlflow_log_metric("prof_underdogs", dog_profit_total)
        self.mlflow_log_metric("prof_home", home_profit_total)
        self.mlflow_log_metric("prof_draw", draw_profit_total)
        self.mlflow_log_metric("prof_away", away_profit_total)
        # self.mlflow_log_metric("profit_val",val_profit)
        # self.mlflow_log_metric("prof_v_favorites",fav_profit_v_total)
        # self.mlflow_log_metric("prof_v_underdogs", dog_profit_v_total)
        # self.mlflow_log_metric("prof_v_home", home_profit_v_total)
        # self.mlflow_log_metric("prof_v_draw", draw_profit_v_total)
        # self.mlflow_log_metric("prof_v_away", away_profit_v_total)

        return season_profit


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=[FutureWarning,DeprecationWarning])

    experiment = "BeatTheBookies"
    df, test_df = get_data(test_season='2019/2020')
    print(df.shape)
    X = df.drop(columns=['FTR','HTR','home_team', 'away_team', 'season', 'date', 'Referee'])
    y = df['under_win']
    models = ['Logistic', 'KNNClassifier', 'RandomForestClassifier','GaussianNB','XGBClassifier','RidgeClasifier', 'SVC']
    # models = ['Logistic', 'RandomForestClassifier','SVC']
    balancers = ['SMOTE', 'ADASYN', 'RandomOversampler', 'RandomUnderSampler', 'NearMiss']
    # balancers = ['SMOTE', 'RandomUnderSampler']
    # for mod in models:
    #     for bal in balancers:
    #         print(mod, bal, ':')
    #         params = dict(upload=True,
    #                       local=False,  # set to False to get data from GCP (Storage or BigQuery)
    #                       gridsearch=False,
    #                       split=True,
    #                       optimize=False,
    #                       X_test = test_df.drop(columns=['FTR','HTR','home_team', 'away_team', 'season', 'date', 'Referee']),
    #                       y_test = test_df['under_win'],
    #                       y_type='single',
    #                       balance=bal,
    #                       bet = 10,
    #                       estimator=mod,
    #                       mlflow=True,  # set to True to log params to mlflow
    #                       experiment_name=experiment,
    #                       pipeline_memory=None,
    #                       feateng=None,
    #                       n_jobs=-1)
    #         t = Trainer(X=X, y=y, **params)
    #         t.train()
    #         t.evaluate()
    params = dict(upload=True,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  split=True,
                  optimize=False,
                  X_test = test_df.drop(columns=['FTR','HTR','home_team', 'away_team', 'season', 'date', 'Referee']),
                  y_test = test_df['under_win'],
                  y_type='single',
                  balance='SMOTE',
                  bet = 10,
                  estimator='LDA',
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment,
                  pipeline_memory=None,
                  feateng=None,
                  n_jobs=-1)
    t = Trainer(X=X, y=y, **params)
    t.train()
    t.evaluate()


