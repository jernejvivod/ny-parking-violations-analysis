import math


class StreamedAggs:
    def __init__(self, column):
        """Compute streamed mean and standard deviation."""

        # current mean, variance and count of processed values
        self.column = column
        self._mean = 0.0
        self._var = 0.0
        self._count = 0

    def __call__(self, acc, row):
        prev_mean = self._mean
        prev_std = self._var

        # if not value signalling unknown/missing value, compute next mean and variance
        self._mean = ((prev_mean * self._count) + x) / (self._count + 1)
        self._var = ((self._var + prev_mean ** 2) * self._count + x ** 2) / (self._count + 1) - self._mean ** 2
        self._count += 1

        return {'mean': prev_mean, 'std': math.sqrt(prev_std if abs(prev_std) > 1.0e-16 else 0.0)}
