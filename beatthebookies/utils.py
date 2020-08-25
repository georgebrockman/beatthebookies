import time

def compute_accuracy(y_pred, y_true):
    wrong = abs(y_pred - y_true)
    print(y_pred[0:10])
    print(y_true[0:10])
    print(wrong[0:10])
    return wrong.sum()




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
