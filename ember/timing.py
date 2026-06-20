import time
from contextlib import contextmanager


@contextmanager
def Timer():
    """Context manager that records elapsed wall-clock time.

    Usage::

        with Timer() as t:
            do_work()
        print(f"Elapsed: {t['elapsed']:.1f}s")
    """
    t: dict = {}
    start = time.perf_counter()
    yield t
    t["elapsed"] = time.perf_counter() - start
