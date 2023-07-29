import pandas as pd

from dataclasses import dataclass


@dataclass
class Dataset():
    frame: pd.DataFrame | list[pd.DataFrame] | list[list[pd.DataFrame]]
    feature_names: list
    target_names: list


def load_combines() -> Dataset:

    url = 'https://www.dropbox.com/scl/fi/4dqcr9sdyc6z91925e0yq/data.xls?dl=1&rlkey=1rlgka6ngn7lpja8869flz1m1'
    frame = pd.read_excel(url, skiprows=2)\
        .pivot_table(values='Значение', index='Время', columns='Описание')
    
    return Dataset(frame=frame, feature_names=list(frame.columns), target_names=None)
