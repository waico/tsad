import pandas as pd 

def value_counts_interval(array,itervals):
    """
    input : np.array, list of values
    retrun : pd.series
    """
    names = [f"до {itervals[0]}"]
    quantity = [len(array[array < itervals[0]])]
    for i in range(len(itervals)-1):
        quantity.append(len(array[(array >= itervals[i]) & (array < itervals[i+1])]))
        names.append(f'c {itervals[i]} до {itervals[i+1]}')
    names += [f"от {itervals[-1]}"]
    quantity += [len(array[array >= itervals[-1]])]
    ts = pd.Series(quantity,index=names)    
    return ts



# def (df,num_points):
#     """ 
#     Посмотреть среднее расстояние между всеми парами точек (сэмлов) 
#     в первых num_points точках
#     в последних num_points точках
#     и одновременно в первых и последжних num_points точках.
#     """
#     import itertools
#     import scipy 
#     array1 = df.iloc[:num_points].values
#     array2 = df.iloc[num_points:int(2*num_points)].values

#     indexes = list(range(len(array1)))
#     pairs = list(set(itertools.permutations(indexes, 2)))
#     list1 = array1[np.array(pairs)[:,0]]
#     list2 = array1[np.array(pairs)[:,1]]
#     print('Claster 1', scipy.spatial.distance.cdist(list1,list2).mean())


#     indexes = list(range(len(array2)))
#     pairs = list(set(itertools.permutations(indexes, 2)))
#     list1 = array2[np.array(pairs)[:,0]]
#     list2 = array2[np.array(pairs)[:,1]]
#     print('Claster 2', scipy.spatial.distance.cdist(list1,list2).mean())

#     common_array = np.concatenate([array1,array2])
#     indexes = list(range(len(common_array)))
#     pairs = list(set(itertools.permutations(indexes, 2)))
#     list1 = common_array[np.array(pairs)[:,0]]
#     list2 = common_array[np.array(pairs)[:,1]]
#     print('Claster common', scipy.spatial.distance.cdist(list1,list2).mean())


        
def split_by_repeated(series,df=None):
    """
    retrun dict with lists of ts whwre keys is unique values
    ts[ts.diff()!=0]  побыстрее будет
    """
    series = series.copy().dropna()
    if len(series.unique())==1:
        result = {series.unique()[0]:[series]}
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

    df = df.dropna(how='all').dropna(axis=1,how='all')
    dts  = df.dropna(how='all').index.to_series().diff()
    if resample_freq is None:
        dts_dist = dts.value_counts()
        if dts_dist[0] > dts_dist[1:].sum():
            resample_freq  = dts_dist.index[0]
        else: 
            print(dts_dist)
            raise Exception("Необходимо самостоятельно обработать функцию так как нет преобладающей частоты дискретизации")
    # print(resample_freq,koef_freq_of_gap )
    # print(koef_freq_of_gap )
    thereshold_gap = pd.Timedelta(resample_freq)*koef_freq_of_gap if thereshold_gap is None else thereshold_gap
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
