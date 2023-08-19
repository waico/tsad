import pandas as pd

from dataclasses import dataclass


@dataclass
class Dataset():
    name: str
    description: str
    task: str
    frame: pd.DataFrame | list[pd.DataFrame] | list[list[pd.DataFrame]]
    feature_names: list
    target_names: list


def load_combines() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of Combines dataset.
    '''
    url = 'https://www.dropbox.com/scl/fi/4dqcr9sdyc6z91925e0yq/data.xls?dl=1&rlkey=1rlgka6ngn7lpja8869flz1m1'
    frame = pd.read_excel(url, skiprows=2)\
        .pivot_table(values='Значение', index='Время', columns='Описание')
    
    name = 'Combines state monitoring'
    description = ''
    task = ''
    
    return Dataset(name=name, description=description, task=task, frame=frame, feature_names=list(frame.columns), target_names=None)

def load_tsad_example() -> Dataset:
    '''
    Loads and slightly preprocesses example df for TSAD example
    '''
    url = 'https://www.dropbox.com/scl/fi/0mw3k1sa5p1bc8qhsdqj0/tsad_example.csv?rlkey=486ytrds74t70m6w5w5dgke68&dl=1'
    frame = pd.read_csv(url,index_col='DT',parse_dates=['DT'])
    
    name = 'TSAD example for Tutorial'
    description = ''
    task = ''
    
    return Dataset(name=name, description=description, task=task, frame=frame, feature_names=list(frame.columns), target_names=None)


def load_skab_teaser() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of SKAB (skoltech anomaly benchmark) teaser.
    '''
    # X
    url='https://drive.google.com/file/d/1Gtz3LJLxoyHLatV_d07Pny5wKHGGsbj3/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    frame = pd.read_csv(url, sep=';', parse_dates=['datetime'])\
        .pivot_table(values='value', index='datetime', columns='id')
    
    # y_test
    y_test = [('2019-07-08 18:39:22', '2019-07-08 18:42:32'), 
            ('2019-07-08 18:44:36', '2019-07-08 18:46:51'), 
            ('2019-07-08 19:06:57', '2019-07-08 19:11:31'), 
            ('2019-07-08 19:14:40', '2019-07-08 19:21:16')]
    
    name = 'SKAB (skoltech anomaly benchmark) teaser'
    description = 'Dataset for process monitoring (changepoint detection) benchmarking. It is just a short version (teaser) of SKAB'
    task = 'Process monitoring (changepoint detection)'
    
    return Dataset(name=name, description=description, task=task, frame=[frame, y_test], feature_names=list(frame.columns), target_names=None)

def load_turbofan_jet_engine() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of NASA Turbofan Jet Engine Data Set.
    '''
    feature_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
    target_names = ['ttf']
    
    # X_train and y_train
    url = 'http://azuremlsamples.azureml.net/templatedata/PM_train.txt'
    frame_train = pd.read_csv(url, sep = ' ', header=None)
    frame_train.drop([26,27], axis=1, inplace=True)
    frame_train.columns = feature_names
    
    # X_test
    url = 'http://azuremlsamples.azureml.net/templatedata/PM_test.txt'
    frame_test = pd.read_csv(url, sep = ' ', header=None)
    frame_test.drop([26,27], axis=1, inplace=True)
    frame_test.columns = feature_names
    
    # y_test
    url = 'http://azuremlsamples.azureml.net/templatedata/PM_truth.txt'
    y_test = pd.read_csv(url, sep = ' ', header=None)
    y_test.drop([1], axis=1, inplace=True)
    y_test.columns = target_names
    
    name = 'NASA Turbofan Jet Engine Data Set'
    description = '''Dataset includes Run-to-Failure simulated data from turbo fan jet engines. In this dataset the goal is to predict the remaining useful life (RUL) of each engine in the test dataset. RUL is equivalent of number of flights remained for the engine after the last datapoint in the test dataset.
    - In train dataset there are 100 engines. The last cycle for each engine represents the cycle when failure had happened.
    - In test dataset there are 100 engines as well. But this time, failure cycle was not provided.'''
    task = 'Remaining useful life prediction'
    
    return Dataset(name=name, description=description, task=task, frame=[frame_train, frame_test, y_test], feature_names=feature_names, target_names=target_names)