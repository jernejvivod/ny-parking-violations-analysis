from streamz import Stream

from ny_parking_violations_analysis.rolling_aggregates.aggregates import StreamedMeanStd

FILE_NAME = "data/Parking_Violations_Issued_-_Fiscal_Year_2022_trunc.csv"


def stream(col: int):
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
    for line in open(FILE_NAME):
        if skip:
            skip = False
            continue
        source.emit(line)
    return r[-1]


def row_to_array(x):
    return x.split(",")


def groupby_count(acc, x):
    if x in acc:
        acc[x] += 1
    else:
        acc[x] = 0
    return acc
