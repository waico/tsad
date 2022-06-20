# косяк, мой подсчет метрик не работает если там нет трушных 1
"""
CDSDSDS
"""

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 
def ts_train_test_split(df, len_seq, 
                     points_ahead=1, gap=0, shag=1, intersection=True,
                     test_size=None,train_size=None, random_state=None, what_to_shuffle='train'):
    """
    A function that splits the time series into train and test subsets  

    Parameters
    ----------
    df : pd.DataFrame
        Array of shape (n_samples, n_features) with pd.timestamp index.
    
    points_ahead : int, default=0
        How many points ahead we predict, reflected in y
         
    gap :  int, default=0
        How many points between train and test. Relatively speaking, 
        if the "right" point of train is t, then the first point of the test 
        is t + gap +1. The parameter is designed to be able to predict 
        one point after a large additional time interval.
    
    shag :  int, default=1.
        Sample generation step. If the first point was t for 
        the 1st sample of the train, then for the 2nd sample of the train 
        it will be t + shag if intersection=True, otherwise the same 
        but without intersections of the series values.

    intersection :  bool, default=True
       The presence of series values (of one point in time) in different 
       samples for the train set and and separately for the test test. 
       The train and the test never have common points.
    
    test_size : float or int or timestamp for df, or list of timestamps, default=0.25. 
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. 
        If int, represents the absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25. *
        If timestamp for df, for X_test we will use set from df[t:] 
        If list of timestamps [t1,t2], for X_test we will use set from df[t1:t2] 
        Can be 0, then it will return the X,y values in X_train, y_train. 
        
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size. 
        If timestamp for df, for X_train we will use set for train from df[:t] 
        If list of timestamps [t1,t2], for X_train we will use set for train from df[t1:t2] 
        Can be 0, then it will return the X,y values in X_test, y_test.
        
    what_to_shuffle: {'nothing', 'all','train'}, str. Default = 'train'. 
        In the case of 'train' we random shuffle only X_train, and y_train. 
        Test samples are unused for the shuffle. Any sample from X_test is later 
        than any sample from X_train. This is also true for respectively
        In case of 'all' in analogy with sklearn.model_selection.train_test_split
        In case of 'nothing' shuffle is not performed.
        
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.*  

    Returns
    -------
    (X_train, X_test, y_train, y_test) : tuple 
        Tuple containing train-test split of inputs
    
    
    Examples
    --------
    >>> X = np.ones((4, 3))
    >>> y = np.ones(4)
    >>> sklearn_template(X, y)
    (z, xmin, xmax)  # this should match the actual output
    """
    
    
      # проблема вклюения правых границ
        
    
    
    # Default settings
    if (train_size is None) and (test_size is None):
        train_size = 0.75
        test_size  = 0.25

    assert len_seq + points_ahead + gap  <= len(df)
    assert all(np.sort(df.index)==np.array(df.index))



    ####### Part I: Вычисление подряд возможных samples
    x_start=0
    x_end= x_start + len_seq
    y_start = x_end + gap 
    y_end = y_start + points_ahead
    if intersection:
        # ради вычислительной нагрузки такой кастыль
        def compute_new_x_start(x_start,y_end,shag):
            return x_start + shag
    else:
        def compute_new_x_start(x_start,y_end,shag):
            return y_end + shag -1
    X = []
    y = []
    while y_end <= len(df):
        X.append(df[x_start:x_end])
        y.append(df[y_start:y_end])

        x_start= compute_new_x_start(x_start,y_end,shag)
        x_end= x_start + len_seq
        y_start = x_end + gap
        y_end = y_start + points_ahead

    ####### Part 2: Выход на train_sample_numbers и test_sample_numbers 

    train_sample_numbers = None 
    test_sample_numbers  = None
    if isinstance(train_size,(pd.Timestamp,list)) or isinstance(test_size,(pd.Timestamp,list)):
        df_vot = pd.DataFrame({'start':[_df.index[0] for _df in X], 'end': [_df.index[-1] for _df in y]}).reset_index()  
        def check_list_assert(my_list):
            for val in my_list:
                assert isinstance(val, pd.Timestamp)
        if isinstance(train_size,(pd.Timestamp,list)):
            if isinstance(train_size,pd.Timestamp):
                t_train_left  = None
                t_train_right = train_size
            elif isinstance(train_size,list):
                check_list_assert(train_size)
                t_train_left = train_size[0]
                t_train_right = train_size[1]
            train_indexes = df_vot.set_index('end').truncate(t_train_left,t_train_right)['index']
            train_sample_numbers = len(train_indexes)

        if isinstance(test_size,(pd.Timestamp,list)):
            if isinstance(test_size,pd.Timestamp):
                t_test_left  = test_size
                t_test_right = None
            elif isinstance(test_size,list):
                check_list_assert(test_size)
                t_test_left = test_size[0]
                t_test_right = test_size[1]

            test_indexes  = df_vot.set_index('start').truncate(t_test_left, t_test_right)['index']
            test_sample_numbers = len(test_indexes)

    if isinstance(train_size,float):
        train_sample_numbers = int(len(X)*train_size)
    if isinstance(test_size,float):
        test_sample_numbers = int(len(X)*test_size)

    if isinstance(train_size,int):
        train_sample_numbers = train_size
    if isinstance(test_size,int):
        test_sample_numbers = test_size

    if train_sample_numbers is None: # due to test_size is not defined 
        train_sample_numbers = len(X) - test_sample_numbers
    if test_sample_numbers is None: # due to test_size is not defined 
        test_sample_numbers = len(X) - train_sample_numbers

    if train_sample_numbers + test_sample_numbers > len(X):
        raise Exception("There is not enough data in df dataset, according to the parameters train_size and test_size")



    ####### Part 2: Shuffle and generation of sets

    ind_train_left  = len(X)-1*(train_sample_numbers+test_sample_numbers) 
    ind_train_right = len(X)-test_sample_numbers
    ind_test_left  = len(X)-test_sample_numbers 
    ind_test_right = len(X) 

    assert ind_test_left>=ind_train_right
    assert (ind_test_right - ind_test_left) +  (ind_train_right - ind_train_left) <= len(df)



    if what_to_shuffle == 'nothing':
        X_train = X[ind_train_left:ind_train_right]
        y_train = y[ind_train_left:ind_train_right]
        X_test = X[ind_test_left:ind_test_right]
        y_test = y[ind_test_left:ind_test_right]

    elif what_to_shuffle == 'train':
        X_test = X[ind_test_left:ind_test_right]
        y_test = y[ind_test_left:ind_test_right] 

        X_train_meta = X[:ind_train_right]
        y_train_meta = y[:ind_train_right]

        indices = np.array(range(len(X_train_meta)))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        indices=indices[:train_sample_numbers]
        X_train  = [X_train_meta[i] for i in indices]
        y_train  = [y_train_meta[i] for i in indices]

    elif what_to_shuffle == 'all':
        indices = np.array(range(len(X)))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]

        X_train = X[ind_train_left:ind_train_right]
        y_train = y[ind_train_left:ind_train_right]
        X_test = X[ind_test_left:ind_test_right]
        y_test = y[ind_test_left:ind_test_right]
    else:
        raise Exception('Choose correct what_to_shuffle')
    return X_train, X_test, y_train, y_test
    
    
        
def split_by_repeated(series,df=None):
    """
    retrun dict with lists of ts whwre keys is unique values
    ts[ts.diff()!=0]  побыстрее будет
    """
    series = series.copy().dropna()
    if len(series.unique())==1:
        result = {series.unique()[0]:series}
    elif len(series.unique())>1:
        result = {uni:[] for uni in series.unique()}
        recent_i=0
        recent_val=series.values[0]
        for i in range(len(series)):
            val = series.values[i]
            if (recent_val == val):
                continue
            else:
                result[recent_val].append(series[recent_i:i])
                recent_i=i
                recent_val = val

        if i == len(series)-1:
            if (recent_val == val):
                result[recent_val].append(series[recent_i:i+1])
            else:
                result[recent_val].append(series[recent_i:i+1])
    else:
        raise NameError('0 series')


    if df is not None:
        new_result = {uni:[] for uni in series.unique()}
        for key in result:
            for i in range(len(result[key])):
                if len(result[key][i]) <=1:
                    continue
                else:
                    new_result[key].append(df.loc[result[key][i].index])
        return new_result
    else:
        return result
        
        
def df2dfs(df,  # Авторы не рекомендуют так делать,
            resample_freq = None, # требования
            thereshold_gap = None, 
            koef_freq_of_gap = 1.2, # 1.2 проблема которая возникает которую 02.09.2021 я написал в ИИ 
            plot = False,
            col = None):
    """
    Функция которая преообратает raw df до требований к входу на DL_AD    
    то есть разбивает df на лист of dfs by gaps 
    
    Не ресемлирует так как это тяжелая задача, но если частота реже чем 
    koef_freq_of_gap of thereshold_gap то это воспринимается как пропуск. 
    Основной посыл: если сигнал приходит чаще, то он не уползает сильно, 
    а значит не приводит к аномалии, а если редко то приводит, поэтому воспри-
    нимается как пропуск. 
    
    plot - очень долго
    
    Parameters
    ----------   
    df : pd.DataFrame
        Исходный временной ряд полностью за всю историю   
        
    resample_freq: pd.TimeDelta (optional, default=None)
        Частота дискретизации временного ряда. 
        Если default то самая частая частота дискретизации. При этом, 
        если нет выраженной частоты вылетит ошибка. 
    thereshold_gap : pd.TimeDelta (optional, default=None)
        Порог периода, превышая который функция будет воспринимать
        данный период как пропуск. 
    koef_freq_of_gap : float or int (optional if thereshold_gap==None,
        default=1.2)  
        thereshold_gap = koef_freq_of_gap * resample_freq
    plot : bool (optional, default=True)
        If true, then отрисуется нарезка
        If false, then не отрисуется нарезка   
    col : int of str (optional, default=True)
        Название или номер колонки для отрисовки
        Если None первая колонка
    Returns
    -------
    dfs : list of pd.DataFrame
        Список времменных рядов без пропусков с относительно стабильной 
        частотой дискретизации. 
    """

    df = df.dropna(how='all').dropna(1,how='all')
    dts  = df.dropna(how='all').index.to_series().diff()
    if resample_freq is None:
        dts_dist = dts.value_counts()
        if dts_dist[0] > dts_dist[1:].sum():
            resample_freq  = dts_dist.index[0]
        else: 
            print(dts_dist)
            raise Exception("Необходимо самостоятельно обработать функцию так как нет преобладающей частоты дискретизации")
    thereshold_gap = resample_freq*koef_freq_of_gap if thereshold_gap is None else thereshold_gap
    gaps = (dts > thereshold_gap).astype(int).cumsum()
    dfs = [df.loc[gaps[gaps==stage].index] for stage in gaps.unique()]
    
    if plot:
        f, ax = plt.subplots()
        if col is None:
            col = df.columns[0]
        else:
            if type(col)==type(int):
                col = df.columns[col]
        for df in dfs:
            df[col].plot(ax=ax)
    return dfs

    