import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # check start time
        start = time.perf_counter()

        # execute func
        value = func(*args, **kwargs)

        # check end time
        end = time.perf_counter()

        elapsed = end - start
        print('Finished {} in {} secs'.format(repr(func.__name__), round(elapsed, 7)))

        # bypass return
        return value

    return  wrapper


class StopWatch():
    """ 스탑워치 해주는 기능 """

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.stop_time  = time.perf_counter()

    def __str__(self):
        return str(round(self.stop_time - self.start_time, 7))




