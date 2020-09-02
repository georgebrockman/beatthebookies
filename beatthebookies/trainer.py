import mlflow
import warnings
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow

from beatthebookies.data import get_data
from beatthebookies.utils import simple_time_tracker, compute_scores, compute_overall_scores
from beatthebookies.encoders import FifaDifferentials, WeeklyGoalAverages, WinPctDifferentials, WeeklyGoalAgAverages, ShotOTPct, HomeAdv
from beatthebookies.bettingstrategy import compute_profit, optimizedhomeprofit
from beatthebookies.gcp import storage_upload
from beatthebookies.params import MODEL_VERSION

from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from memoized_property import memoized_property

from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import optimizers, regularizers
# from tensorflow.keras.layers import Embedding, Conv1D, Dense, Flatten, SimpleRNN, Conv2D, MaxPooling1D,Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.layers import BatchNormalization

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

from tempfile import mkdtemp
from collections import Counter
from termcolor import colored



# warnings.filterwarnings("ignore", category=FutureWarning)


MLFLOW_URI = "https://mlflow.lewagon.co/"
myname="Chris_Westerman"
EXPERIMENT_NAME = f"[UK][London][{myname}] BeatTheBookies"

#mlflow.sklearn.autolog()


class Trainer(object):

    ESTIMATOR = 'logistic'
    rs = RandomUnderSampler(random_state=0)

    def __init__(self, X, y, X_test=None, y_test=None, **kwargs):
        self.pipeline = None
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.local = kwargs.get("local", True)
        self.grid = kwargs.get("gridsearch", False)
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.y_type = self.kwargs.get('y_type', 'single')
        self.bet = self.kwargs.get('bet', 10)
        self.gridsearch = self.kwargs.get("gridsearch", False)

        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=15)
            self.X_train, self.y_train = self.balancing(self.X_train, self.y_train)
        else:
            self.X_train, self.y_train = self.balancing(self.X, self.y)

        self.experiment_name = kwargs.get("experiment_name", EXPERIMENT_NAME)
        # self.log_kwargs_params()
        # self.model_params = None

    # def log_kwargs_params(self):
    #       if self.mlflow:
    #             self.mlflow_log_param('balance', self.kwargs.get('balance', 'balance'))
    #             self.mlflow_log_param('gridsearch', self.kwargs.get('gridsearch', False))
    #             self.mlflow_log_param('bet', self.kwargs.get('bet', 10))


    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        # self.mlflow_log_param("model", estimator)
        # added both regressions for predicting scores and classifier for match outcomes
        # elif estimator == 'Linear':
        #     model = LinearRegression()
        # elif estimator == 'RandomForestRegressor':
        #     model = RandomForestRegressor()
        # elif estimator == 'Lasso':
        #     model = Lasso()
        # elif estimator == "Ridge":
        #     model = Ridge()
        # elif estimator == "GBM":
        #     model = GradientBoostingRegressor()
        # elif estimator == "KNNRegressor":
        #     model = KNeighborsRegressor()
        if estimator == 'GaussianNB':
            model = GaussianNB()
        # elif estimator == 'LDA':
        #     self.model_params = {'solver': ['lsqr','eigen'],  #note svd does not run with shrinkage and models using it will be tuned separately
        #                           'n_components': [1.0,2.0,3.0,4.0,5.0]}
        #     model = LinearDiscriminantAnalysis()
        # elif estimator == "xgboost":
        #     model = XGBRegressor()
        # classification models
        if estimator == 'Logistic':
            self.model_params = {'C': np.arange(0.001,1000)}
            #model = LogisticRegression(C=435.0009999999999)
            model = LogisticRegression()
        elif estimator == 'LDA':
            model = LinearDiscriminantAnalysis()
        elif estimator == 'RandomForestClassifier':
            self.model_params = {'bootstrap': [True, False],
                                 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                 'max_features': ['auto', 'sqrt'],
                                 'min_samples_leaf': [1, 2, 4],
                                 'min_samples_split': [2, 5, 10],
                                 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            #model = RandomForestClassifier(n_estimators=1800, n_jobs=-1,max_depth=100,min_samples_split=5,bootstrap=False)
            model = RandomForestClassifier()
        elif estimator == "RidgeClassifier":
            self.model_params = {"alpha": np.arange(0.001,1000)}
            # model = RidgeClassifier(alpha=71.00099999999999)
            model = RidgeClassifier()
            # model = GridSearchCV(estimator=grid, param_grid=dict(alpha=alphas))
        elif estimator == "KNNClassifier":
            self.model_params = {"leaf_size": range(1,10),
                                 "n_neighbors": range(1,10),
                                 "p":[1.0,2.0]}
            # model = KNeighborsClassifier(leaf_size=6,n_neighbors=6,p=1.0) #positive results
            model = KNeighborsClassifier()
            # model = GridSearchCV(knn, hyperparameters, cv=10)
        elif estimator == "XGBClassifier":
            self.model_params = {'max_depth': range(2, 20, 2),
                                 'n_estimators': range(60, 220, 40),
                                 'learning_rate': [0.3, 0.1, 0.01, 0.05],
                                 'min_child_weight': [1.0, 3.0, 5.0],
                                 'gamma': [1.0, 3.0, 5.0]}
            # model = XGBClassifier(max_depth=18,n_estimators=60,learning_rate=0.05,min_child_weight=5,gamma=3.0) #positive results
            model = XGBClassifier()
            # model = GridSearchCV(XGB, param_grid=params_1, cv=5)
        elif estimator == "SVC":
            model = SVC(kernel='poly', probability=True)
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
            self.model_params = {'C': [0.001,0.01,0.1,1,10,100,1000]}
            model = LogisticRegression()

        estimator_params = self.kwargs.get("estimator_params", {})
        if estimator != "Sequential":
            model.set_params(**estimator_params)
        return model

    def set_pipeline(self):
        feateng_steps = self.kwargs.get('feateng', ['fifadiff','windiff','goaldiff','homeadv','shototpct','goalagdiff'])

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

        for bloc in feateng_blocks:
            if bloc[0] not in feateng_steps:
              feateng_blocks.remove(bloc)

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


    def add_gridsearch(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator
        {'rgs__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
          'rgs__max_features' : ['auto', 'sqrt'],
          'rgs__max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        """
        # Here to apply ramdom search to pipeline, need to follow naming "rgs__paramname"

        #mlflow.sklearn.autolog()

        params = {"rgs__" + k: v for k, v in self.model_params.items()}
        self.pipeline = RandomizedSearchCV(estimator=self.pipeline, param_distributions=params,
                                           n_iter=10,
                                           cv=2,
                                           verbose=1,
                                           random_state=42,
                                           n_jobs=-1)
                                           #pre_dispatch=None)


    @simple_time_tracker
    def train(self):
        tic = time.time()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        self.set_pipeline()
        if self.gridsearch:
            self.add_gridsearch()
        if self.kwargs.get("estimator", self.ESTIMATOR) == 'Sequential':
            self.pipeline.fit(self.X_train, self.y_train,  rgs__validation_split=0.2, rgs__shuffle=True, rgs__epochs=300,
                        rgs__batch_size=32, rgs__verbose=1, rgs__callbacks=[es])
        else:
            #self.pipeline.fit(self.X_train, self.y_train)
            pipelinefit = self.pipeline.fit(self.X_train, self.y_train)
            best_estimator = pipelinefit.best_estimator_
            print(best_estimator)
            self.mlflow_log_param("best_estimator",best_estimator)
            return pipelinefit


    def evaluate(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        bet = self.bet
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")

        if self.split:
            y_val_pred = self.pipeline.predict_proba(self.X_val)
            y_val_pred = y_val_pred[:,1:].reshape((len(y_val_pred),))
        # y_test_pred = self.pipeline.predict(self.X_test)
        y_test_pred = self.pipeline.predict_proba(self.X_test) #.reshape((380,))
        y_test_pred = y_test_pred[:,1:].reshape((len(y_test_pred),))

        def predict_threshold(x):
            if x['pct'] >= self.kwargs.get('threshold', 0.5):
                return 1
            return 0

        if self.y_type == 'pct':
            convert = pd.DataFrame({'pct': y_test_pred})
            convert['predict'] = convert.apply(lambda x: predict_threshold(x), axis=1)
            positives = convert.predict.sum()
            # stake = optimizedhomebet(self.X_test, convert, self.y_test, bankroll=100)
            # profit = optimizedhomeprofit(self.X_test, convert['pct'], self.y_test, bankroll=100):
            kelly_criterion = optimizedhomeprofit(self.X_test, convert.pct.to_numpy(), self.y_test, bankroll=10)
            season_profit, _, dog_profit_total, _, _, _ = compute_profit(self.X_test, convert.predict.to_numpy(), self.y_test, bet)
            scores = compute_overall_scores(convert.predict, self.y_test)
            # val
            convert = pd.DataFrame({'pct': y_val_pred})
            convert['predict'] = convert.apply(lambda x: predict_threshold(x), axis=1)
            val_positives = convert.predict.sum()
            kelly_criterion = optimizedhomeprofit(self.X_val, convert.pct.to_numpy(), self.y_val, bankroll=10)
            val_model_profit, _, val_dog_profit_total, _, _, _ = compute_profit(self.X_val, convert.predict.to_numpy(), self.y_val, bet)
            val_scores = compute_overall_scores(convert.predict, self.y_val)

        if self.y_type == 'label':
            val_scores = compute_overall_scores(y_val_pred, self.y_val)
            scores = compute_overall_scores(y_test_pred, self.y_test)
            val_model_profit, _, val_dog_profit_total, val_home_profit_total, _, _ = compute_profit(self.X_val, y_val_pred, self.y_val, bet)
            season_profit, _, dog_profit_total, home_profit_total, _, _ = compute_profit(self.X_test, y_test_pred, self.y_test, bet)
            val_positives = y_val_pred.sum()
            positives = y_test_pred.sum()

        # self.mlflow_log_metric("best_estimator",best_estimator)
        run_id = self.mlflow_run().info.run_id
        self.mlflow_log_param("model", estimator,run_id)
        self.mlflow_log_param('balance', self.kwargs.get('balance', 'balance'),run_id)
        self.mlflow_log_param('gridsearch', self.kwargs.get('gridsearch', False),run_id)
        self.mlflow_log_param('bet', self.kwargs.get('bet', 10),run_id)


        self.mlflow_log_metric("val_f1",val_scores[1],run_id)
        self.mlflow_log_metric("val_accuracy",val_scores[0],run_id)
        self.mlflow_log_metric("val_recall",val_scores[2],run_id)
        self.mlflow_log_metric("val_f1",val_scores[3],run_id)
        self.mlflow_log_metric('val_picked', val_positives,run_id)
        self.mlflow_log_metric('val_prof_underdogs', val_dog_profit_total,run_id)
        self.mlflow_log_metric('val_model_profits', val_model_profit,run_id)


        self.mlflow_log_metric('test_picked', positives, run_id)
        self.mlflow_log_metric("test_precision",scores[1],run_id)
        self.mlflow_log_metric("test_accuracy",scores[0],run_id)
        self.mlflow_log_metric("test_recall",scores[2],run_id)
        self.mlflow_log_metric("profit_model",season_profit,run_id)
        self.mlflow_log_metric("prof_underdogs", dog_profit_total,run_id)

        # self.mlflow_log_metric("prof_favorites",fav_profit_total)
        self.mlflow_log_metric("prof_home", home_profit_total)
        # self.mlflow_log_metric("prof_draw", draw_profit_total)
        # self.mlflow_log_metric("prof_away", away_profit_total)
        # self.mlflow_log_metric("profit_val",val_profit)
        # self.mlflow_log_metric("prof_v_favorites",fav_profit_v_total)
        # self.mlflow_log_metric("prof_v_underdogs", dog_profit_v_total)
        self.mlflow_log_metric("val_home_profit", val_home_profit_total)
        # self.mlflow_log_metric("prof_v_draw", draw_profit_v_total)
        # self.mlflow_log_metric("prof_v_away", away_profit_v_total)

        return season_profit

    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        joblib.dump(self.pipeline, 'beatthebookies/data/model.joblib')
        print(colored("model.joblib saved locally", "green"))

        if not self.local:
            storage_upload(model_version=MODEL_VERSION)

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

    #@memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value, run_id):
        self.mlflow_client.log_param(run_id, key, value)

    def mlflow_log_metric(self, key, value, run_id):
        self.mlflow_client.log_metric(run_id, key, value)



if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=[FutureWarning,DeprecationWarning])

    cols = ['H_ATT', 'A_ATT', 'H_MID', 'A_MID', 'H_DEF', 'A_DEF', 'H_OVR', 'A_OVR',
        'home_t_total_wins','away_t_total_wins', 'stage','home_t_total_goals','away_t_total_goals',
        'home_t_home_goals','home_t_home_goals_against','away_t_away_goals','away_t_away_goals_against',
        'home_t_prev_home_matches', 'away_t_prev_away_matches', 'home_t_total_shots', 'home_t_total_shots_ot',
        'away_t_total_shots', 'away_t_total_shots_ot', 'home_t_total_goals_against','away_t_total_goals_against', 'WHH', 'WHA', "WHD",
        'home_w', 'away_w', 'draw', 'winning_odds']

    experiment = "BeatTheBookies"
    # df, test_df = get_data(test_season='2019/2020')
    # X = df.drop(columns=['FTR','HTR','home_team', 'away_team', 'season', 'date', 'Referee'])
    # y = df['home_w']
    # models = ['Logistic', 'KNNClassifier', 'RandomForestClassifier','GaussianNB','XGBClassifier','RidgeClasifier', 'SVC', 'LDA']
    # balancers = ['SMOTE', 'ADASYN', 'RandomOversampler', 'RandomUnderSampler', 'NearMiss']
    # models = ['Logistic', 'RandomForestClassifier','SVC','KNNClassifier']
    models = ['XGBClassifier']
    balancers = ['SMOTE', 'RandomUnderSampler']

    # for mod in models:
    #     for bal in balancers:
    #         print(mod, bal, ':')
    params = dict(upload=True,
                  test_season='2019/2020',
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  split=True,
                  y_type='pct',
                  balance='SMOTE',
                  bet = 10,
                  threshold=0.5,
                  estimator='SVC',
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment,
                  pipeline_memory=None,
                  feateng=None,
                  n_jobs=-1)
    df, test_df = get_data(**params)
    # X = df.drop(columns=['FTR','HTR','home_team', 'away_team', 'season', 'date', 'Referee'])
    X = df[cols]
    y = df['home_w']
    # X_test = test_df.drop(columns=['FTR','HTR','home_team', 'away_team', 'season', 'date', 'Referee'])
    X_test = test_df[cols]
    y_test = test_df['home_w']
    t = Trainer(X=X, y=y, X_test=X_test, y_test=y_test, **params)
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()

    # params = dict(upload=True,
    #               local=False,  # set to False to get data from GCP (Storage or BigQuery)
    #               gridsearch=False,
    #               split=True,
    #               optimize=False,
    #               X_test = test_df.drop(columns=['FTR','HTR','home_team', 'away_team', 'season', 'date', 'Referee']),
    #               y_test = test_df['under_win'],
    #               y_type='pct',
    #               balance='SMOTE',
    #               bet = 100,
    #               threshold = 0.85,
    #               estimator='SVC',
    #               mlflow=True,  # set to True to log params to mlflow
    #               experiment_name=experiment,
    #               pipeline_memory=None,
    #               feateng=None,
    # #               n_jobs=-1)
    # t = Trainer(X=X, y=y, **params)
    # print(colored("############  Training model   ############", "red"))
    # t.train()
    # print(colored("############  Evaluating model ############", "blue"))
    # t.evaluate()
    # print(colored("############   Saving model    ############", "green"))
    # t.save_model()


