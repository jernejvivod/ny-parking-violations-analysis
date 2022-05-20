from streamz import Stream

from ny_parking_violations_analysis.rolling_aggregates.aggregates import StreamedAggs

FILE_NAME = 'Parking_Violations_Issued_-_Fiscal_Year_2022'


def stream(col: int, windows_size: int):
    r = list()
    source = Stream()
    source.scatter().map(row_to_array).pluck(col).accumulate(
        StreamedAggs, start=({})
    ).sink(r.append)

    for line in open(FILE_NAME):
        source.emit(line)
    return r[-1]


def row_to_array(x):
    return x.split(",")
