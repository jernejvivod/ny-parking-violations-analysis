from fileinput import filename
import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd

def groupby_count(df: dd.DataFrame, col: str, amt: int=10) -> None:
    return (df.groupby(col)
            .agg({'Summons Number': 'count'})
            .compute()
            .sort_values(['Summons Number'], ascending=False)
            .reset_index(level=0)
            .rename(columns={'Summons Number': 'Ticket Count'})
            .head(amt))

def plot_bar(df: pd.DataFrame, x: str, y: str, filename: str) -> None:
    df.plot(x=x, y=y, kind='bar')
    plt.tight_layout()
    plt.savefig(fname=f'ny-parking-violations-analysis/img/{filename}')
