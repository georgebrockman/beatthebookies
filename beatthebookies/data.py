import itertools
import os
import pandas as pd
import numpy as np
from google.cloud import storage
from beatthebookies.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH
from beatthebookies.utils import simple_time_tracker


@simple_time_tracker
def get_data(test_season='2019/2020', league='EPL', local=False, **kwargs):
    client = storage.Client()
    if local:
        root_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(root_dir, 'beatthebookies/', 'data/')
        # file = csv_path + league + '.csv'
        file = csv_path + 'premiertotals.csv'
    else:
        file = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)

    df = pd.read_csv(file)
    test_df = df[df['season'] == test_season]
    df = df[df['season'] != test_season]


    return df, test_df




if __name__ == "__main__":
    print('hey')








