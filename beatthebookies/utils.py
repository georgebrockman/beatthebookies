import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score



def compute_overall_scores(y_pred,y_true):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true,y_pred, average='weighted',zero_division=0)
    rec = recall_score(y_true,y_pred, average='weighted')
    f1 = f1_score(y_true,y_pred, average='weighted',zero_division=0)
    scores =  [acc,pre,rec,f1]
    print(acc)
    return scores

def compute_scores(y_pred,y_true):
    precision,recall,fscore,support=score(y_true,y_pred,zero_division=0)
    scores = [precision,recall,fscore,support]
    return scores

    # for each game see whether our model predicted correct
    # if yes, multiply betting amount times ratio
    # compute overall profit


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
