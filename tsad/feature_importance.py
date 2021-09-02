def feature_importance(residuals, analysis_type="collective", date_from=None, date_till=None, weigh=True):
    """Feature importance calculation

    Parameters
    ----------
    residuals : pandas.DataFrame()

    analysis_type : str, "single"/"collective", "single" by default

    date_from : str in format 'yyyy-mm-dd HH:MM:SS', None by default

    date_till : str in format 'yyyy-mm-dd HH:MM:SS', None by default

    weigh : boolean, True by default
        If analysis_type == "collective".

    Returns
    -------
    data : pandas.DataFrame().
    """
    if date_from is None:
        start = 0
    if date_till is None:
        end = -1
    data = residuals[date_from:date_till].abs().copy()

    if (analysis_type == "collective") & (weigh == False):
        data = data.div(data.sum(axis=1), axis=0) * 100
        return pd.DataFrame(data.mean(), columns=['Feature importance, %']).T
    elif (analysis_type == "collective") & (weigh == True):
        data = data.mean().div(data.mean().sum(), axis=0) * 100
        return pd.DataFrame(data, columns=['Feature importance, %']).T
    elif analysis_type == "single":
        return data.div(data.sum(axis=1), axis=0) * 100   
