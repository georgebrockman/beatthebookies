import itertools
import os
import pandas as pd
import numpy as np
import sqlite3

from beatthebookies.utils import simple_time_tracker



def get_prem_league():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(root_dir, 'beatthebookies/', 'data/')
    file = csv_path + 'premiertotals.csv'
    df = pd.read_csv(file)

    return df




if __name__ == "__main__":
    print('hey')








