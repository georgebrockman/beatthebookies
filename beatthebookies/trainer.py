import mlflow
import warnings
import time
import pandas as pd
from beatthebookies.data import get_data
from beatthebookies.utils import simple_time_tracker, compute_scores, compute_overall_scores
# from beatthebookies.encoders import CustomNormaliser, CustomStandardScaler


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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
#from beatthebookies.encoders import CustomNormaliser, CustomStandardScaler
from tempfile import mkdtemp
from beatthebookies.bettingstrategy import compute_profit

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
            model = LogisticRegression(solver='newton-cg')
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

        self.pipeline = Pipeline(steps=[
          ('scale', RobustScaler()),
          ('rgs', self.get_estimator())])


    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        #x_train.drop(columns=['WHH', 'WHD', 'WHA'])
        self.pipeline.fit(self.X_train.drop(columns=['WHH', 'WHD', 'WHA']), self.y_train)


    def evaluate(self,bet):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(self.X_test.drop(columns=['WHH', 'WHD', 'WHA']))
        overall_scores = compute_overall_scores(y_pred,self.y_test)
        scores = compute_scores(y_pred,self.y_test)
        self.mlflow_log_metric("accuracy",overall_scores[0])

        self.mlflow_log_metric("precision",overall_scores[1])
        self.mlflow_log_metric("precision_home",scores[0][0])
        self.mlflow_log_metric("precision_away",scores[0][1])
        self.mlflow_log_metric("precision_draw",scores[0][2])

        self.mlflow_log_metric("recall",overall_scores[2])
        self.mlflow_log_metric("recall_home",scores[1][0])
        self.mlflow_log_metric("recall_away",scores[1][1])
        self.mlflow_log_metric("recall_draw",scores[1][2])

        self.mlflow_log_metric("f1",overall_scores[3])
        self.mlflow_log_metric("f1_home",scores[2][0])
        self.mlflow_log_metric("f1_away",scores[2][1])
        self.mlflow_log_metric("f1_draw",scores[2][2])

        self.mlflow_log_metric("support_home",scores[3][0])
        self.mlflow_log_metric("support_away",scores[3][1])
        self.mlflow_log_metric("support_draw",scores[3][2])

        profit = compute_profit(self.X_test[['WHH', 'WHA','WHD']],y_pred,self.y_test,bet)
        self.mlflow_log_metric("profit",profit)

        return scores


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
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # seasons = ['2009/2010', '2010/2011', '2011/2012', '2012/2013',
    #          '2013/2014', '2014/2015', '2015/2016']
    experiment = "BeatTheBookies"
    params = dict(season='2015/2016',
                  full=True,
                  upload=True,
                  local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  optimize=False,
                  estimator="KNNClassifier",
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment,
                  pipeline_memory=None,
                  feateng=None,
                  n_jobs=-1)
    df = get_data(**params)
    bet = 10
    df.dropna(inplace=True)
    print(df.shape)
    X = df.drop(columns=['id', 'season', 'date', 'stage', 'home_team_goal', 'away_team_goal', 'home_team', 'away_team', 'home_w', 'away_w', 'draw'])
    y = df[['home_w', 'away_w', 'draw']]
    t = Trainer(X=X, y=y, **params)
    t.train()
    t.evaluate(bet)

