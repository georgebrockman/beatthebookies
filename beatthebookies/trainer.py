import mlflow
import time
import pandas as pd
# import warnings
from beatthebookies.data import get_data
from beatthebookies.utils import compute_accuracy, simple_time_tracker

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


# warnings.filterwarnings("ignore", category=FutureWarning)


MLFLOW_URI = "https://mlflow.lewagon.co/"
myname="Chris_Westerman"
EXPERIMENT_NAME = f"[UK][London][{myname}] BeatTheBookies"


class Trainer(object):

    ESTIMATOR = 'logistic'

    def __init__(self, X, y, **kwargs):
        self.pipeline = None
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)
        self.X = X
        self.y = y
        if self.split:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                  test_size=0.3, random_state=15)
        self.experiment_name = kwargs.get("experiment_name", EXPERIMENT_NAME)
        self.model_params = None


    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        self.mlflow_log_param("model", estimator)
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
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        return model



        return model

    def set_pipeline(self):

        # features_encoder = ColumnTransformer([
        #     ('distance', DistanceTransformer(), list(DIST_ARGS.values())),
        #     ('time_features', pipe_time_features, ['pickup_datetime']),
        #     ('distance_to_center', pipe_d2center,list(DIST_ARGS.values())),
        #     ('geohash', pipe_geohash, list(DIST_ARGS.values()))
        # ])


        self.pipeline = Pipeline(steps=[('rgs', self.get_estimator())])


    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(self.X_test)
        acc = compute_accuracy(y_pred, self.y_test)
        self.mlflow_log_metric("accuracy", acc)
        return round(acc, 3)

     def evaluateall(self):
          if self.pipeline is None:
              raise ("Cannot evaluate an empty pipeline")
          y_pred = self.pipeline.predict(self.X_test)
          acc = compute2_accuracy(y_pred, self.y_test)
          pre = compute_precision(y_pred,self.y_test)
          rec = compute_recall(y_pred,self.y_test)
          self.mlflow_log_metric("total accuracy", acc[0])
          self.mlflow_log_metric("home_w accuracy", acc[1])
          self.mlflow_log_metric("away_w accuracy", acc[2])
          self.mlflow_log_metric("draw accuracy", acc[3])

          self.mlflow_log_metric("precision", pre)
          self.mlflow_log_metric("recall", rec)
          return 'Accuracy:{} || Prcision: {} || Recall: {}'.format(round(acc[0], 3), round(pre,3), round(rec,3))


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
    seasons = ['2009/2010', '2010/2011', '2011/2012', '2012/2013',
             '2013/2014', '2014/2015', '2015/2016']
    experiment = "BeatTheBookies"
    params = dict(season='2008/2009',
                  upload=True,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  optimize=False,
                  estimator="Logistic",
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment,
                  pipeline_memory=None,
                  feateng=None,
                  n_jobs=-1)
    df = get_data(**params)
    for year in seasons:
        x = get_data(season=year)
        df = pd.concat([df, x], axis=0)
    X = df.drop(columns=['id', 'season', 'date', 'home_team_goal', 'away_team_goal', 'home_team', 'away_team', 'home_w', 'away_w', 'draw'])
    y = df['home_w']
    t = Trainer(X=X, y=y, **params)
    t.train()
    t.evaluate()

