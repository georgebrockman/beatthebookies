import mlflow
import warnings
import time
import pandas as pd
from beatthebookies.data import get_data
from beatthebookies.utils import simple_time_tracker, compute_scores, compute_overall_scores
from beatthebookies.encoders import FifaDifferentials, WeeklyGoalAverages, WinPctDifferentials, WeeklyGoalAgAverages


from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from tempfile import mkdtemp
from beatthebookies.bettingstrategy import compute_profit
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

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
        self.X = X
        self.y = y
        self.y_type = self.kwargs.get('y_type', 'single')

        if self.y_type == 'single':
            self.le = LabelEncoder()
            self.le.fit(self.y)
            print(self.le.classes_)
            self.y = self.le.transform(self.y)


        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=15)

        self.experiment_name = kwargs.get("experiment_name", EXPERIMENT_NAME)
        self.model_params = None

        self.X_train, self.y_train = self.balancing()

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        # self.mlflow_log_param("model", estimator)
        # added both regressions for predicting scores and classifier for match outcomes
        if estimator == 'Logistic':
            model = LogisticRegression()
        elif estimator == 'Linear':
            model = LinearRegression()
        elif estimator == 'RandomForestClassifier':
            model = RandomForestClassifier()
        elif estimator == 'RandomForestRegressor':
            model = RandomForestRegressor()
        elif estimator == 'Lasso':
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "KNNClassifier":
            model = KNeighborsClassifier()
        elif estimator == "KNNRegressor":
            model = KNeighborsRegressor()
        elif estimator == 'GaussianNB':
            model = GaussianNB()
        elif estimator == "xgboost":
            model = XGBRegressor()
        elif estimator == "SVC":
            model = SVC()
            self.model_params = { 'kernel': [['linear', 'poly', 'rbf']]}
        else:
            model = LogisticRegression()
        estimator_params = self.kwargs.get("estimator_params", {})
        # self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        return model

    def set_pipeline(self):
       #  memory = self.kwargs.get("pipeline_memory", None)
       #  feateng_steps = self.kwargs.get("feateng", ["normalise", "standardise"])
       #  if memory:
       #      memory = mkdtemp()
       #  # adding temporay row choices for each transformer to see if it works
       #  feateng_blocks = ColumnTransformer([
       #    ('normalise', CustomNormaliser(), )
       #    ('standardise', CustomStandardScaler(), )
       #    ])

       #  pipe_standard = make_pipeline(CustomStandardScaler())
       #  pipe_normalise = make_pipeline(CustomNormaliser())

       #  features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None,\
       #   remainder="drop")
       #  # features_encoder = ColumnTransformer([
       #  #     ('distance', DistanceTransformer(), list(DIST_ARGS.values())),
       #  #     ('time_features', pipe_time_features, ['pickup_datetime']),
       #  #     ('distance_to_center', pipe_d2center,list(DIST_ARGS.values())),
       #  #     ('geohash', pipe_geohash, list(DIST_ARGS.values()))
       #  # ])

       #  # Filter out some bocks according to input parameters
       #  for bloc in feateng_blocks:
       #      if bloc[0] not in feateng_steps:
       #          feateng_blocks.remove(bloc)
        # pipe_scale = ColumnTransformer(StandardScaler())

          # 'home_t_total_goals', 'home_t_total_shots','home_t_total_goals_against', 'home_t_total_shots_against', 'away_t_total_goals',
          # 'away_t_total_goals_against', 'away_t_total_shots', 'away_t_total_shots_against', 'home_t_total_wins', 'home_t_total_losses',
          # 'away_t_total_wins', 'away_t_total_losses'
        pipe_fifadiff = make_pipeline(FifaDifferentials(), RobustScaler())
        pipe_winpct = make_pipeline(WinPctDifferentials(), StandardScaler())
        pipe_avggoal = make_pipeline(WeeklyGoalAverages(), StandardScaler())
        pipe_avggoal_ag = make_pipeline(WeeklyGoalAgAverages(), StandardScaler())

        feateng_blocks = [('fifadiff', pipe_fifadiff, ['H_ATT', 'A_ATT', 'H_MID', 'A_MID', 'H_DEF', 'A_DEF', 'H_OVR', 'A_OVR']),
                          ('windiff', pipe_winpct, ['home_t_total_wins','away_t_total_wins', 'stage']),
                          ('goaldiff', pipe_avggoal, ['home_t_total_goals','away_t_total_goals', 'stage']),
                          ('goalagdiff', pipe_avggoal_ag, ['home_t_total_goals_against','away_t_total_goals_against', 'stage'])
                         ]

        features_encoder = ColumnTransformer(feateng_blocks, n_jobs=None, remainder="drop")


        self.pipeline = Pipeline(steps=[
          ('features', features_encoder),
          ('rgs', self.get_estimator())])


    def balancing(self):
        ### OVERSAMPLERS
        balance = self.kwargs.get("balance", "SMOTE")
        if balance == "SMOTE":
          # Create new samples without making any disticntion between easy and hard samples to be classified using K-nearest neighbor
          X_train, y_train = SMOTE().fit_resample(self.X_train, self.y_train)
          print(Counter(y_train))
          return X_train, y_train
        elif balance == "ADASYN":
          # Create new samples next to the original samples which are wrongly classified by using K-Nearest neighbor
          X_train, y_train = ADASYN().fit_resample(self.X_train, self.y_train)
          print(Counter(y_train))
          return X_train, y_train
        elif balance == "RandomOversampler":
          # Duplicating some of the original samples of the minority class
          X_train, y_train = RandomOverSampler(random_state=0).fit_resample(self.X_train, self.y_train)
          print(Counter(y_train))
          return X_train, y_train

       ### UNDERSAMPLERS
        if balance == "RandomUnderSampler":
          # balances the data by randomly selecting a subset of data for the targeted classes
          X_train, y_train = RandomUnderSampler(random_state=0).fit_resample(self.X_train, self.y_train)
          print(Counter(y_train))
          return X_train, y_train
        if balance == "CLusterCentroids":
          # Selects samples based on k-nearest neighbor
          X_train, y_train = ClusterCentroids(random_state=0).fit_resample(self.X_train, self.y_train)
          print(Counter(y_train))
          return X_train, y_train
        if balance == "NearMiss":
          # Allows to select 3 different rules of selecting samples based on k-neearest neighbors (version 1,2,3)
          X_train, y_train = NearMiss(version=1).fit_resample(self.X_train, self.y_train)
          print(Counter(y_train))
          return X_train, y_train
        else:
          return X_train, y_train




    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)


    def evaluate(self):
        bet = self.bet = self.kwargs.get("bet", 10)
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(self.X_val)
        overall_scores = compute_overall_scores(y_pred,self.y_val)
        scores = compute_scores(y_pred,self.y_val)
        # self.mlflow_log_metric("accuracy",overall_scores[0])

        # self.mlflow_log_metric("precision",overall_scores[1])
        # self.mlflow_log_metric("precision_home",scores[0][0])
        # self.mlflow_log_metric("precision_away",scores[0][1])
        # self.mlflow_log_metric("precision_draw",scores[0][2])

        # self.mlflow_log_metric("recall",overall_scores[2])
        # self.mlflow_log_metric("recall_home",scores[1][0])
        # self.mlflow_log_metric("recall_away",scores[1][1])
        # self.mlflow_log_metric("recall_draw",scores[1][2])

        # self.mlflow_log_metric("f1",overall_scores[3])
        # self.mlflow_log_metric("f1_home",scores[2][0])
        # self.mlflow_log_metric("f1_away",scores[2][1])
        # self.mlflow_log_metric("f1_draw",scores[2][2])

        # self.mlflow_log_metric("support_home",scores[3][0])
        # self.mlflow_log_metric("support_away",scores[3][1])
        # self.mlflow_log_metric("support_draw",scores[3][2])

        profit, fav_profit_total, dog_profit_total, home_profit_total, draw_profit_total, away_profit_total = compute_profit(self.X_val, y_pred, self.y_val, bet)

        # self.mlflow_log_metric("profit_model",profit)
        # self.mlflow_log_metric("prof_favorites",fav_profit_total)
        # self.mlflow_log_metric("prof_underdogs", dog_profit_total)
        # self.mlflow_log_metric("prof_home", home_profit_total)
        # self.mlflow_log_metric("prof_draw", draw_profit_total)
        # self.mlflow_log_metric("prof_away", away_profit_total)

        return scores


    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    experiment = "BeatTheBookies"
    params = dict(upload=True,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  optimize=False,
                  y_type='single',
                  balance="RandomUnderSampler",
                  bet = 10,
                  estimator="logistic",
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment,
                  pipeline_memory=None,
                  feateng=None,
                  n_jobs=-1)
    df, test_df = get_data(test_season='2019/2020')
    print(df.shape)
    X = df.drop(columns=['FTR'])
    y = df['FTR']
    t = Trainer(X=X, y=y, **params)
    t.train()
    t.evaluate()

