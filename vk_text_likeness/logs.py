import inspect

import time

from collections import defaultdict

_method_time_logs = defaultdict(list)


def log_method_begin():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller_name = "{}: {}".format(calframe[1].filename.split('/')[-1], calframe[1].function)
    _method_time_logs[caller_name].append(time.time())
    print("{}: begin".format(caller_name))


def log_method_end():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller_name = "{}: {}".format(calframe[1].filename.split('/')[-1], calframe[1].function)
    if caller_name in _method_time_logs:
        logs = _method_time_logs[caller_name]
        if len(logs) > 0:
            print("{}: end ({}s)".format(caller_name, time.time() - logs[-1]))
            logs.pop()
    else:
        print("{}: end".format(caller_name))
