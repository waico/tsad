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
                     test_size=None,train_size=None, random_state=None, shuffle=True,stratify=None):
    """
    Функция которая разбивает временной ряд на трейн и тест выборки 
    
    Временной ряд здесь это вообще вся история
    Функционал позволяет разбивать ....

    Parameters
    ----------
    df : pd.DataFrame
        Array of shape (n_samples, n_features) with timestamp index.
    len_seq : int 
        Длина 
    flag : bool (optional, default=True)
        If true, then do one thing.
        If false, then do another thing.
    f : callable (optional, default=None)
        Call-back function.  If not specified, then some other function
        will be used
    **kwargs :
        Additional keyword arguments will be passed to name_of_function
    
    Returns
    -------
    z : ndarray
        result of shape (n_samples,).  Note that here we use "ndarray" rather
        than "array_like", because we assure we'll return a numpy array.
    
    TODO
    --------
    t-test of timestamp
    
    Examples
    --------
    >>> X = np.ones((4, 3))
    >>> y = np.ones(4)
    >>> sklearn_template(X, y)
    (z, xmin, xmax)  # this should match the actual output
    """

#    """
#	df - требование, но если тебе не хочется то просто сделай np.array на выходе и все
#    Разбить временные ряды на трейн и тест
#    len_seq- количество времменых точек в трейн
#    points_ahead - количество времменых точек в прогнозе
#    gap - расстояние между концом трейна и началом теста
#    intersection - если нет, то в выборке нет перескающих множеств (временнызх моментов)
#    shag - через сколько прыгаем
#    train_size - float от 0 до 1
#    
#    return list of dfs
#    
#    """
    #TODO требования к входным данным прописать
    #TODO переписать энергоэффективно чтобы было
    #TODO пока временные характеристики int_ами пора бы в pd.TimdDelta
    # нет индексов 
    assert len_seq + points_ahead + gap + 1 <= len(df)
    how='seq to seq'   

# -------------------------------------------------------  
#             
# -------------------------------------------------------  


    x_start=0
    x_end= x_start + len_seq
    y_start = x_end + gap +1
    y_end = y_start + points_ahead
    
    if intersection:
        # ради вычислительной нагрузки такой кастыль
        def compute_new_x_start(x_start,y_end,shag):
            return x_start + shag
    else:
        def compute_new_x_start(x_start,y_end,shag):
            return y_end + shag
    
    X = []
    y = []
    while y_end <= len(df):
        X.append(df[x_start:x_end])
        y.append(df[y_start:y_end])
        
        x_start= compute_new_x_start(x_start,y_end,shag)
        x_end= x_start + len_seq
        y_start = x_end + gap +1
        y_end = y_start + points_ahead
          
    
    if (test_size==0) | (len(X)==1):
        indices = np.array(range(len(X)))
        #             np.random.seed(random_state)
        if shuffle:
            print(indices)
            np.random.shuffle(indices)
            print(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]
        return X,[],y,[]
    else:
        return train_test_split(X,y, 
                                test_size=test_size, 
                                train_size=train_size, 
                                random_state=random_state, 
                                shuffle=shuffle, 
                                stratify=stratify
                               )
						   
						   
						   
class Loader:
    def __init__(self, X,y, batch_size,shuffle=True):
        if shuffle==True:
            indices = np.array(range(len(X)))
            np.random.shuffle(indices)
            self.X = X[indices]
            self.y = y[indices]
        else:
            self.X = X
            self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        self.i = - self.batch_size
        return self

    def __next__(self):
        if self.i+self.batch_size < len(self.X):
            self.i+=self.batch_size
            return self.X[self.i:self.i+self.batch_size], self.y[self.i:self.i+self.batch_size]
        elif self.i+2*self.batch_size < len(self.X):        
            return self.X[self.i:], self.y[self.i:]
        else:
            raise StopIteration
            
    def __len__(self):
        return len(np.arange(0,len(self.X),self.batch_size))
		        
        
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

    