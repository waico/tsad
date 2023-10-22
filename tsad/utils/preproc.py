import pandas as pd 

def value_counts_interval(array,itervals):
    """
    Returns a pandas Series containing the count of values in the 
    input array that fall within each interval.

    Parameters:
    ----------
    array : numpy.ndarray | list of values
        Input array of values.
    intervals (list): 
        List of interval boundaries. The first interval is defined 
        as values less than the first boundary, and the last interval 
        is defined as values greater than or equal to the last boundary.

    Returns:
    -------
    ts : pandas.Series
        A Series containing the count of values in the input array 
        that fall within each interval.
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



        
def split_by_repeated(series,df=None):
    """
    Splits a pandas series into sub-series based on repeated values.

    Parameters:
    ----------
    series : pandas.Series
        The series to be split.
    df (, optional): pandas.DataFrame. Defaults is None.
        The dataframe to be used to retrieve the original rows. 

    Returns:
    -------

    dict: A dictionary where the keys are the unique values in the series and the values are lists of sub-series.
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
    Function that splits df into a list of dfs by gaps. That is it makes raw df 
    satisfying to the input requirements with the lack of gaps and different 
    frequencies of discretization.
    Does not resample as it is a heavy task, but if the frequency is less than 
    koef_freq_of_gap of thereshold_gap, it is perceived as a skip. 
    The main idea: if the signal comes more often, then it does not slip too much, 
    and therefore does not lead to anomalies, but if it is rare, it leads to anomalies, 
    so it is perceived as a skip. 
    
    plot - very long
    
    Parameters
    ----------   
    df : pd.DataFrame
        The original time series for the entire history.   
        
    resample_freq: pd.TimeDelta (optional, default=None)
        The frequency of time series discretization. 
        If default, then the most frequent frequency of discretization. 
        If there is no pronounced frequency, an error will occur. 
    thereshold_gap : pd.TimeDelta (optional, default=None)
        The threshold period, exceeding which the function will perceive 
        this period as a skip. 
    koef_freq_of_gap : float or int (optional if thereshold_gap==None,
        default=1.2)  
        thereshold_gap = koef_freq_of_gap * resample_freq
    plot : bool (optional, default=False)
        Plot the cut, but it is need very long time. 
        If true, then the cut will be drawn.
        If false, then the cut will not be drawn.   
    col : int of str (optional, default=True)
        The name or number of the column to draw.
        If None, the first column is used.
    Returns
    -------
    dfs : list of pd.DataFrame
        A list of time series without gaps with a relatively stable 
        frequency of discretization. 
    """

    df = df.dropna(how='all').dropna(axis=1,how='all')
    dts  = df.dropna(how='all').index.to_series().diff()
    if resample_freq is None:
        dts_dist = dts.value_counts()
        if dts_dist[0] > dts_dist[1:].sum():
            resample_freq  = dts_dist.index[0]
        else: 
            print(dts_dist)
            raise Exception("It is necessary to process the function yourself since there is no prevailing sampling frequency")
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