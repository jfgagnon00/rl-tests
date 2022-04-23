from time import perf_counter


class ScopedPerfCounter:
    def __init__(self):
        self._start = 0.0
        self._stop = 0.0
        self.reset()

    def reset(self):
        self._total = 0.0

    def start(self):
        self._start = perf_counter()

    def stop(self):
        self._stop = perf_counter()
        self._total += self.delta()

    def delta(self):
        return self._stop - self._start

    def deltaMs(self):
        return self.delta() * 1000.0

    def total(self):
        return self._total

    def totalMs(self):
        return self.total() * 1000.0

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
