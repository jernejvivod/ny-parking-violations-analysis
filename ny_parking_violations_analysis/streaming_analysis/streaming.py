from typing import List

import numpy
import pandas as pd
from sklearn.cluster import Birch
from streamz import Stream

from .aggregates import StreamedMeanStd

BATCH_SIZE = 1000


def stream(col: int, file_name: str) -> dict:
    r = list()
    stats = StreamedMeanStd()
    source = Stream()
    (
        source.map(row_to_array)
            .pluck(col)
            .accumulate(groupby_count, start=({}))
            .map(stats)
            .sink(r.append)
    )

    skip = True
    for line in open(file_name):
        if skip:
            skip = False
            continue
        source.emit(line)
    return pd.DataFrame(r[-1], index=[0])


def stream_clustering(cols: List[int], file_name: str) -> dict:
    r = list()
    source = Stream()
    (
        source.map(row_to_array)
            .pluck(cols)
            .accumulate(
            stream_cluster, start=({'batch': [], 'cluster': Birch(n_clusters=10), 'encoding': {}})
        )
            .sink(r.append)
    )

    skip = True
    for line in open(file_name):
        if skip:
            skip = False
            continue
        source.emit(line)
    return r[-1]


def stream_cluster(acc, x):
    encoded = []
    for i in range(len(x)):
        if i in acc['encoding']:
            if x[i] not in acc['encoding'][i]:
                acc['encoding'][i][x[i]] = len(acc['encoding'][i])
        else:
            acc['encoding'][i] = dict()
            acc['encoding'][i][x[i]] = 0
        encoded.append(acc['encoding'][i][x[i]])
    encoded = numpy.array(encoded, dtype=numpy.float64)

    if len(acc['batch']) < BATCH_SIZE:
        acc['batch'].append(encoded)
    else:
        acc['cluster'].partial_fit(acc['batch'])
        acc['batch'] = [encoded]
    return acc


def row_to_array(x):
    return x.split(',')


def groupby_count(acc, x):
    if x in acc:
        acc[x] += 1
    else:
        acc[x] = 0
    return acc
