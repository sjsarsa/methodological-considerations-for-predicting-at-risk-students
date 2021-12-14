from functools import wraps
from time import time

import warnings
warnings.filterwarnings("ignore")

def get_var_values(cls):
    return [path for var, path in cls.__dict__.items()
            if not var.startswith('_')]


def get_var_names(cls):
    return [path for var, path in cls.__dict__.items()
            if not var.startswith('_')]


def get_vars_as_dict(cls):
    return {var: path for var, path in cls.__dict__.items()
            if not var.startswith('_')}


def print_elapsed(msg, start_time, format='min'):
    if format not in ('min', 's'): raise ValueError('format must be "min" or "s"')
    elapsed = time() - start_time
    if format == 'min': elapsed /= 60
    print('{} in {:.2f} {}'.format(msg, elapsed, format))


class Time:
    def __init__(self, start_msg, end_msg=None, format='min'):
        if format not in ('min', 's'): raise ValueError('format must be "min" or "s"')

        self.start_msg = start_msg
        self.end_msg = end_msg
        self.format = format

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kw):
            start = time()
            print(self.start_msg + '...')
            res = func(*args, **kw)
            print_elapsed(self.end_msg or self.start_msg + ' finished', start, self.format)
            return res

        return wrapper


class ForEachETA:
    def __init__(self, iterable, start_msg, end_msg=None, mode='linear'):
        self.iterable = iterable
        self.start_msg = start_msg
        self.end_msg = end_msg
        self.mode = mode

    def __call__(self, func):
        @wraps(func)
        def wrapper():
            start_time = time()
            n = len(self.iterable)
            print(self.start_msg)
            A = []
            for i, item in enumerate(self.iterable):
                iter_start_time = time()
                A.append(func(item))
                # print('\r', end='')
                p_done = ((i + 1) / n) * 100
                eta = (n - i - 1) * (time() - iter_start_time) / 60
                print(f'\r{p_done:.4}% {self.start_msg} done, ETA: {eta:.2f} min', end='')
            print_elapsed(self.end_msg or f'{self.start_msg} finished', start_time)
            return A

        return wrapper
