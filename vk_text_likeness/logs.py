import inspect

import time

_method_time_logs = {}


def log_method_begin():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller_name = "{}: {}".format(calframe[1].filename.split('/')[-1], calframe[1].function)
    _method_time_logs[caller_name] = time.time()
    print("{}: begin".format(caller_name))


def log_method_end():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller_name = "{}: {}".format(calframe[1].filename.split('/')[-1], calframe[1].function)
    if caller_name in _method_time_logs:
        print("{}: end ({}s)".format(caller_name, time.time() - _method_time_logs[caller_name]))
    else:
        print("{}: end".format(caller_name))
    del _method_time_logs[caller_name]
