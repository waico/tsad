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
        names.append(f'{itervals[i]} до {itervals[i+1]}')
    names += [f"от {itervals[-1]}"]
    quantity += [len(array[array >= itervals[-1]])]
    ts = pd.Series(quantity,index=names)    
    return ts