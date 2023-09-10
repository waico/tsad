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

def list_of_datasets():
    '''
    Shows the list of available for import datasets.
    
    Returns
    -------
    list_of_datasets : dict
    '''
    list_of_datasets = {'Combines state monitoring':'load_combines()',
                        'SKAB (skoltech anomaly benchmark) teaser':'load_skab_teaser()',
                        'SKAB (skoltech anomaly benchmark)':'load_skab()',
                        'NASA Turbofan Jet Engine Data Set':'load_turbofan_jet_engine()',
                        'TEP (Tennessee Eastman process)':'load_tep()',
                        'Pressurized Water Reactor (PWR) Dataset for Fault Detection':'load_pwr_anomalies()',
                        'NPP Power Transformer RUL':'load_transformer_rul()'}
    return list_of_datasets

def load_combines() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of Combines dataset.
    
    Returns
    -------
    list_of_datasets : list
    
    References
    ----------
    L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/lbfgsb.html
    '''
    url = 'https://www.dropbox.com/scl/fi/4dqcr9sdyc6z91925e0yq/data.xls?dl=1&rlkey=1rlgka6ngn7lpja8869flz1m1'
    frame = pd.read_excel(url, skiprows=2)\
        .pivot_table(values='Значение', index='Время', columns='Описание')
    
    name = 'Combines state monitoring'
    description = ''
    task = ''
    target_names=None
    
    return Dataset(name=name, description=description, task=task, frame=frame, feature_names=list(frame.columns), target_names=target_names)

def load_skab_teaser() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of SKAB (skoltech anomaly benchmark) teaser.
    
    Returns
    -------
    Dataset
        A dataset object with the folowing structure:
            name : str
            description : str
            task : str
            frame: list[pd.DataFrame]
            feature_names : list
            target_names : list
    
    References
    ----------
    SKAB - Skoltech Anomaly Benchmark | teaser
        Iurii Katser and Viacheslav Kozitsin.
        https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab-teaser
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
    target_names=None
    
    return Dataset(name=name, description=description, task=task, frame=[frame, y_test], feature_names=list(frame.columns), target_names=target_names)

def load_skab() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of SKAB (skoltech anomaly benchmark).
    
    Returns
    -------
    Dataset
        A dataset object with the folowing structure:
            name : str
            description : str
            task : str
            frame: pd.DataFrame
            feature_names : list
            target_names : list
    
    References
    ----------
    Skoltech anomaly benchmark (skab).
        Katser, Iurii D., and Vyacheslav O. Kozitsin. Kaggle (2020).
        https://www.kaggle.com/dsv/1693952
    '''
    url='https://drive.google.com/file/d/1_aeGB3M3CNSEqYPxHPuGK0ju6hMuJUoK/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    frame = pd.read_csv(url, sep=',', parse_dates=['datetime'])
    frame.set_index(['experiment', 'datetime'], inplace=True)
    
    name = 'SKAB (skoltech anomaly benchmark)'
    description = 'Dataset for process monitoring (changepoint detection) benchmarking'
    task = 'Process monitoring (changepoint detection)'
    feature_names = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 
                     'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
    target_names = ['anomaly', 'changepoint']
    
    return Dataset(name=name, description=description, task=task, frame=frame, feature_names=feature_names, target_names=target_names)

def load_turbofan_jet_engine() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of NASA Turbofan Jet Engine Data Set.
    
    Returns
    -------
    Dataset
        A dataset object with the folowing structure:
            name : str
            description : str
            task : str
            frame: list[pd.DataFrame]
            feature_names : list
            target_names : list
    
    References
    ----------
    Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation
        A. Saxena, K. Goebel, D. Simon, and N. Eklund. in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
        https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
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

def load_tep() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of TEP (Tennessee Eastman process) dataset.
    
    Returns
    -------
    Dataset
        A dataset object with the folowing structure:
            name : str
            description : str
            task : str
            frame: pd.DataFrame
            feature_names : list
            target_names : list
    
    References
    ----------
    Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation
        Professor Richard Braatz. Large Scale Systems Research Laboratory.
        https://github.com/YKatser/CPDE/tree/master/TEP_data
    '''
    url='https://drive.google.com/file/d/1zQq2TDKv0fBvXrDwkr9S08k3a3RNPDHO/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    frame = pd.read_csv(url, sep=',').rename(columns={'Unnamed: 0':'index'})
    frame.set_index(['experiment', 'index'], inplace=True)
    
    name = 'TEP (Tennessee Eastman process)'
    description = 'Each training data file contains 480 rows and 52 columns and each testing data file contains 960 rows and 52 columns.  An observation vector at a particular time instant is given by x=[XMEAS(1), XMEAS(2), ..., XMEAS(41), XMV(1), ..., XMV(11)]^T where XMEAS(n) is the n-th measured variable and XMV(n) is the n-th manipulated variable.'
    task = 'Outlier detection'
    target_names=None
    
    return Dataset(name=name, description=description, task=task, frame=frame, feature_names=list(frame.columns), target_names=target_names)

def load_pwr_anomalies() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of Pressurized Water Reactor (PWR) Dataset.
    
    Returns
    -------
    Dataset
        A dataset object with the folowing structure:
            name : str
            description : str
            task : str
            frame: pd.DataFrame
            feature_names : list
            target_names : list
    
    References
    ----------
    Pressurized Water Reactor (PWR) Dataset for Fault Detection
        ENGR. MUSHFIQUR RASHID KHAN
        https://www.kaggle.com/datasets/prottoymushfiq/pressurized-water-reactor-abnormality-dataset
    '''
    url='https://drive.google.com/file/d/1JjPzjqU9QWoFvrlEizoJvqTT6n0OVgNN/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    frame = pd.read_csv(url, sep=',', index_col='Readings')
    
    name = 'Pressurized Water Reactor (PWR) Dataset for Fault Detection'
    description = 'Our collected dataset is benchmark data in case of reactor abnormalities detection with labels. There are 267 readings from 14 sensors of three categories: a temperature sensor, pressure sensor, and vibration sensor (including ionization chamber, accelerometer, and relative displacement sensors). This particular dataset can be utilized in the case of unsupervised abnormality detection.'
    task = 'Anomaly detection'
    target_names=None
    
    return Dataset(name=name, description=description, task=task, frame=frame, feature_names=list(frame.columns), target_names=target_names)

def load_transformer_rul() -> Dataset:
    '''
    Loads and slightly preprocesses raw data of NPP Power Transformer.
    
    Returns
    -------
    Dataset
        A dataset object with the folowing structure:
            name : str
            description : str
            task : str
            frame: list[pd.DataFrame]
            feature_names : list
            target_names : list
    
    References
    ----------
    Machine Learning Methods for Anomaly Detection in Nuclear Power Plant Power Transformers.
        Katser, Iurii, et al. arXiv preprint arXiv:2211.11013 (2022).
    '''
    url='https://drive.google.com/file/d/1_aeGB3M3CNSEqYPxHPuGK0ju6hMuJUoK/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    frame = pd.read_csv(url, sep=',', parse_dates=['datetime'])
    frame.set_index(['experiment', 'datetime'], inplace=True)
    
    # X_train
    url='https://drive.google.com/file/d/1NSbmnIGE5foofxOCd-tQIlbnhAjSjvZX/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    X_train = pd.read_csv(url, sep=',').rename(columns={'Unnamed: 0':'id', 'Unnamed: 1':'time point'})
    X_train.set_index(['id', 'time point'], inplace=True)
    
    # X_test
    url='https://drive.google.com/file/d/1cb7uxJ3wmAZsGyzK1ZhW_S_sUU_JGqjJ/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    X_test = pd.read_csv(url, sep=',').rename(columns={'Unnamed: 0':'id', 'Unnamed: 1':'time point'})
    X_test.set_index(['id', 'time point'], inplace=True)
    
    # y_train
    url='https://drive.google.com/file/d/17akYhUR6R2qhc2PU8OCm9Alrp4sKxitC/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    y_train = pd.read_csv(url, sep=',', index_col='id')
    
    # y_test
    url='https://drive.google.com/file/d/1-NUEm1yiAEdr42JXBbIwXx0tauWyGVvA/view?usp=share_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    y_test = pd.read_csv(url, sep=',', index_col='id')
    
    name = 'NPP Power Transformer RUL'
    description = '''Dataset for Determining the Remaining Useful Life of Transformers. It is necessary to create a mathematical model that will determine RUL by the final 420 points. The period between time points is 12 hours.'''
    task = 'Remaining useful life prediction'
    feature_names = ['H2', 'CO', 'C2H4', 'C2H2']
    target_names = ['predicted']
    
    return Dataset(name=name, description=description, task=task, frame=[X_train, X_test, y_train, y_test], feature_names=feature_names, target_names=target_names)