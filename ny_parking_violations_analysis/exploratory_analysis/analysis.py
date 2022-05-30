from datetime import datetime

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd


def groupby_count(df: dd.DataFrame, col: str, amt: int = 10) -> dd.DataFrame:
    return (
        df.groupby(col)
            .agg({'Summons Number': 'count'})
            .compute()
            .sort_values('Summons Number', ascending=False)
            .reset_index(level=0)
            .rename(columns={'Summons Number': 'Ticket Count'})
            .head(amt)
    )


def plot_bar(df: pd.DataFrame, x: str, y: str, filename: str) -> None:
    print(df.head())
    ax = df.plot(x=x, y=y, kind='bar')

    # Skip crowded ticks, works only for datetime column x
    if len(df[x]) > 15:
        xticks = ax.xaxis.get_major_ticks()
        labels = list(map(lambda x: datetime.strftime(x, '%d-%m-%Y'), df[x]))
        for i, tick in enumerate(xticks):
            if i % 10 != 0:
                tick.label1.set_visible(False)
        ax.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig(fname=f'ny-parking-violations-analysis/results/{filename}')


def map_datestr_to_dt(df: dd.DataFrame) -> dd.DataFrame:
    df['Issue Date'] = df['Issue Date'].map_partitions(
        pd.to_datetime, format='%m/%d/%Y', meta=(None, 'datetime64[s]')
    )
    return df
