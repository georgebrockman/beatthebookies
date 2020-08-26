import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score



def compute_overall_scores(y_pred,y_true):
    acc = accuracy_score(y_true,y_pred)
    pre = precision_score(y_true,y_pred, average='weighted')
    rec = recall_score(y_true,y_pred, average='weighted')
    f1 = f1_score(y_true,y_pred, average='weighted')
    scores =  [acc,pre,rec,f1]
    return scores

def compute_scores(y_pred,y_true):
    precision,recall,fscore,support=score(y_true,y_pred)
    scores = [precision,recall,fscore,support]
    print(scores)
    return scores




def compute_accuracy(y_pred, y_true):
    wrong = abs(y_pred - y_true)
    print(y_pred[0:10])
    print(y_true[0:10])
    print(wrong[0:10])
    return wrong.sum()

def compute2_accuracy(y_pred, y_true):
    acc = accuracy_score(y_true,y_pred)
    return acc

def compute_precision(y_pred,y_true):
    #pre = precision_score(y_true,y_pred, average='weighted')
    precision,recall,fscore,support=score(y_true,y_pred,average='weighted')
    scores = [precision,recall,fscore,support]
    print(scores)
    return score

def compute_recall(y_pred,y_true):
    score = recall_score(y_true,y_pred, average='weighted')
    return score

def compute_f1(y_pred,y_true):
    score = f1_score(y_true,y_pred, average='weighted')
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
