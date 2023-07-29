"""
Данный модуль описывает варианты работы с трейн тест разбиением последовательности. 
"""

import numpy as np
import pandas as pd


 
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



def ts_train_test_split_dfs( dfs, len_seq,
                            points_ahead=1, gap=0, shag=1, intersection=True,
                            test_size=None,train_size=None, random_state=None, what_to_shuffle='train'):
        """
        Вспомогательная функция, избавляющая от дубляжа
        """

        if (type(dfs) == pd.core.series.Series) | (type(dfs) == pd.core.frame.DataFrame):
            df = dfs.copy() if type(dfs) == pd.core.frame.DataFrame else pd.DataFrame(dfs)
            assert len_seq + points_ahead + gap - 1 <= len(df)
            X_train, X_test, y_train, y_test = ts_train_test_split(df=dfs,
                                                                       len_seq=len_seq,
                                                                       points_ahead=points_ahead,
                                                                       gap=gap,
                                                                       shag=shag,
                                                                       intersection=intersection,
                                                                       test_size=test_size,
                                                                       train_size=train_size,
                                                                       random_state=random_state,
                                                                       what_to_shuffle=what_to_shuffle,
                                                                       # потому что потом нужно в основном итераторе
                                                                       )

        elif type(dfs) == type(list()):
            # уже все pd.DataFrame
            _df = pd.concat(dfs, ignore_index=True)
            X_train, X_test, y_train, y_test = [], [], [], []
            _k = 0
            for df in dfs:
                if ((type(df) == pd.core.series.Series) | (type(df) == pd.core.frame.DataFrame)) == False:
                    raise NameError('Type of dfs is unsupported')
                if not (len_seq + points_ahead + gap + 1 <= len(df)):
                    _k += 1
                    continue
                _X_train, _X_test, _y_train, _y_test = ts_train_test_split(df, len_seq,
                                                                               points_ahead=points_ahead,
                                                                               gap=gap,
                                                                               shag=shag,
                                                                               intersection=intersection,
                                                                               test_size=test_size,
                                                                               train_size=train_size,
                                                                               random_state=random_state,
                                                                               what_to_shuffle=what_to_shuffle,
                                                                             )
                X_train += _X_train
                X_test += _X_test
                y_train += _y_train
                y_test += _y_test

            print(
                f'Пропущено {_k} датастов, из-за того что saples слишком малов в датасете. (len_seq + points_ahead + gap -1 <= len(df))')

        else:
            raise NameError('Type of dfs is unsupported')

        return [X_train, X_test, y_train, y_test]
    


    