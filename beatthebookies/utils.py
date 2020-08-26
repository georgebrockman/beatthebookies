import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_accuracy(y_true, y_pred):
    wrong = abs(y_pred - y_true)
    print(y_pred[0:10])
    print(y_true[0:10])
    print(wrong[0:10])
    return wrong.sum()

def compute2_accuracy(y_pred, y_true):
    total = accuracy_score(y_true,y_pred)
    home = accuracy_score(y_true['home_w'],y_pred['home_w'])
    away = accuracy_score(y_true['away_w'],y_pred['away_w'])
    draw = accuracy_score(y_true['draw'],y_pred['draw'])
    scores = [total,home,away,draw]
    return scores

def compute_precision(y_pred,y_true):
    score = precision_score(y_true,y_pred)
    return score

def compute_recall(y_pred,y_true):
    score = recall_score(y_true,y_pred)
    return score

################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
