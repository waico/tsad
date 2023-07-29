import pandas as pd
import numpy as np

def filter_detecting_boundaries(detecting_boundaries):
    """
    [[t1,t2],[],[t1,t2]] -> [[t1,t2],[t1,t2]]
    [[],[]] -> []
    """
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple)!=0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries
    return detecting_boundaries


def single_detecting_boundaries(true_series,true_list_ts, prediction,
                                portion, numenta_time,
                                anomaly_window_destenation,intersection_mode):
    """
    Extrcaty detecting_boundaries from series or list of timestamps
    """


    if (true_series is not None) and (true_list_ts is not None):
        raise Exception('Choose the ONE type')
    elif true_series is not None:
        true_timestamps = true_series[true_series==1].index
    elif true_list_ts is not None:
        if len(true_list_ts) == 0:
            return [[]]
        else:
            true_timestamps = true_list_ts
    else:
        raise Exception('Choose the type')
    #
    detecting_boundaries=[]
    td = pd.Timedelta(numenta_time) if numenta_time is not None else \
                pd.Timedelta((prediction.index[-1]-prediction.index[0])/(len(true_timestamps)+1)*portion)  
    for val in true_timestamps:
        if anomaly_window_destenation == 'lefter':
            detecting_boundaries.append([val - td, val])
        elif anomaly_window_destenation == 'righter':
            detecting_boundaries.append([val, val + td])
        elif anomaly_window_destenation == 'center':
            detecting_boundaries.append([val - td/2, val + td/2])
        else:
            raise('choose anomaly_window_destenation')
            
            
    # block for resolving intersection problem:
    # важно не ошибиться, и всегда следить, чтобы везде правая граница далее
    # не включалась, иначе будет пересечения окон    
    if len(detecting_boundaries)==0:
        return detecting_boundaries
        
    new_detecting_boundaries = detecting_boundaries.copy()
    for i in range(len(new_detecting_boundaries)-1):
        if new_detecting_boundaries[i][1] >= new_detecting_boundaries[i+1][0]:
            print(f'Intersection of scoring windows{new_detecting_boundaries[i][1], new_detecting_boundaries[i+1][0]}')
            if intersection_mode == 'cut left window':
                new_detecting_boundaries[i][1] = new_detecting_boundaries[i+1][0]
            elif intersection_mode == 'cut right window':
                new_detecting_boundaries[i+1][0] = new_detecting_boundaries[i][1]
            elif intersection_mode == 'cut both':
                _a  = new_detecting_boundaries[i][1]
                new_detecting_boundaries[i][1] = new_detecting_boundaries[i+1][0]
                new_detecting_boundaries[i+1][0] = _a
            else:
                raise Exception("choose the intersection_mode")
    detecting_boundaries = new_detecting_boundaries.copy()            
    return detecting_boundaries
    
    
def check_errors(my_list):
    """
    Check format of input true data 
    
    Parameters
    ----------    
    my_list - uniform format of true (See evaluating.evaluating)
    
    Returns
    ----------
    mx : depth of list, or variant of processing 
    """
    assert isinstance(my_list, list)
    mx = 1
#     ravel = []
    level_list = {}
    
    def check_error(my_list):
        return not (\
                    (all(isinstance(my_el, list) for my_el in my_list)) or\
                    (all(isinstance(my_el, pd.Series) for my_el in my_list)) or\
                    (all(isinstance(my_el, pd.Timestamp) for my_el in my_list))
                   )
    
    def recurse(my_list,level=1):
        nonlocal mx
        nonlocal level_list

        if check_error(my_list):
            raise Exception(f"Non uniform data format in level {level}: {my_list}")
            
        
        if level not in level_list.keys():
            level_list[level] = [] # for checking format  
            
        for my_el in my_list:
            level_list[level].append(my_el)
            if isinstance(my_el, list):
                mx = max([mx,level+1])
                recurse(my_el,level+1)

###########################               
    recurse(my_list)
    for level in level_list:
        if check_error(level_list[level]):
            raise Exception(f"Non uniform data format in level {level}: {my_list}")  

    if 3 in level_list: 
        for el in level_list[2]:
            if not( (len(el)==2) or (len(el)==0) ):
                raise Exception(f"Non uniform data format in level {2}: {my_list}") 

    return mx
    
    
def extract_cp_confusion_matrix(detecting_boundaries, prediction,point=0,binary=False):
    """
    prediction: pd.Series
    
    point=None for binary case
    Returns
    ----------
    dict: TPs: dict of numer window of [t1,t_cp,t2]
    FPs: list of timestamps
    FNs: list of of numer window
    """
    _detecting_boundaries = []
    for couple in detecting_boundaries.copy():
        if len(couple)!=0:
            _detecting_boundaries.append(couple)
    detecting_boundaries = _detecting_boundaries
    
    times_pred = prediction[prediction.dropna()==1].sort_index().index
    
    my_dict = {}
    my_dict['TPs'] = {}
    my_dict['FPs'] = []
    my_dict['FNs'] = []
    
    
    if len(detecting_boundaries)!=0:
        my_dict['FPs'].append(times_pred[times_pred<detecting_boundaries[0][0]]) # left
        for i in range(len(detecting_boundaries)):
            times_pred_window = times_pred[(times_pred >= detecting_boundaries[i][0]) &
                                           (times_pred <= detecting_boundaries[i][1])]
            times_prediction_in_window = prediction[detecting_boundaries[i][0]:detecting_boundaries[i][1]].index
            if len(times_pred_window)==0:
                if not binary:
                    my_dict['FNs'].append(i)
                else:
                    my_dict['FNs'].append(times_prediction_in_window)
            else:
                my_dict['TPs'][i]= [detecting_boundaries[i][0],
                                    times_pred_window[point] if not binary else times_pred_window, # внимательно
                                    detecting_boundaries[i][1]]
                if binary:
                    my_dict['FNs'].append(times_prediction_in_window[~times_prediction_in_window.isin(times_pred_window)])
            if len(detecting_boundaries)>i+1:
                my_dict['FPs'].append(times_pred[(times_pred > detecting_boundaries[i][1] ) & \
                                                 (times_pred < detecting_boundaries[i+1][0])])
                                                 
        my_dict['FPs'].append(times_pred[times_pred> detecting_boundaries[i][1]]) #right
    else:
        my_dict['FPs'].append(times_pred)
    
    # кастыль далее на 12 строчек
    if len(my_dict['FPs'])>1:
        my_dict['FPs'] = np.concatenate(my_dict['FPs'])
    elif len(my_dict['FPs'])==1:
        my_dict['FPs'] = my_dict['FPs'][0]
    if len(my_dict['FPs'])==0: # not elif specially 
        my_dict['FPs'] = []

    if binary:
        if len(my_dict['FNs'])>1:
            my_dict['FNs'] = np.concatenate(my_dict['FNs']) 
        elif len(my_dict['FNs'])==1:
            my_dict['FNs'] = my_dict['FNs'][0]
        if len(my_dict['FNs'])==0: # not elif specially 
            my_dict['FNs'] = []
    return my_dict
