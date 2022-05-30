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


def stream_cluster(cols: list[int], file_name: str) -> dict:
    r = list()
    source = Stream()
    (
        source.map(row_to_array)
        .pluck(cols)
        .accumulate(
            stream_cluster, start=({'batch': [], 'cluster': Birch(n_clusters=None)})
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
    if len(acc['batch']) < BATCH_SIZE:
        acc['batch'].append(x)
    else:
        acc['cluster'].partial_fit(x)
        acc['batch'] = [x]
    return acc


def row_to_array(x):
    return x.split(',')


def groupby_count(acc, x):
    if x in acc:
        acc[x] += 1
    else:
        acc[x] = 0
    return acc
