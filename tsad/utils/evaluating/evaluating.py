"""
CDSDSDS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



from .univariate_funcs import confusion_matrix, single_average_delay, single_evaluate_nab
from .src import single_detecting_boundaries, check_errors, extract_cp_confusion_matrix


def evaluating(true, prediction,
               metric='nab',
               numenta_time=None,
               portion=None,
               anomaly_window_destenation='lefter',
               clear_anomalies_mode = True,
               intersection_mode='cut right window',
               table_of_coef=None,
               scale_func = "improved",
               scale_koef=1,               
               plot_figure=False,
               verbose=True
               ):
    """
    Parameters
    ----------
    true: variants:
        or: if one  dataset : pd.Series with binary int labels (1 is 
        anomaly, 0 is not anomaly);
        
        or: if one  dataset : list of pd.Timestamp of true labels, or [] 
        if haven't labels ;
        
        or: if one  dataset : list of list of t1,t2: left and right 
        detection, boundaries of pd.Timestamp or [[]] if haven't labels
        
        or: if many datasets: list (len of number of datasets) of pd.Series 
        with binary int labels;
        
        or: if many datasets: list of list of pd.Timestamp of true labels, or
        true = [ts,[]] if haven't labels for specific dataset;
        
        or: if many datasets: list of list of list of t1,t2: left and right 
        detection boundaries of pd.Timestamp;        
        If we haven't true labels for specific dataset then we must insert 
        empty list of labels: true = [[[]],[[t1,t2],[t1,t2]]]. 
        
        __True labels of anomalies or changepoints.
        It is important to have appropriate labels (CP or 
        anomaly) for corresponding metric (See later "metric")
        
    prediction: variants:
        or: if one  dataset : pd.Series with binary int labels
        (1 is anomaly, 0 is not anomaly);
        
        or: if many datasets: list (len of number of datasets) 
        of pd.Series with binary int labels.
        
        __Predicted labels of anomalies or changepoints.
        It is important to have appropriate labels (CP or 
        anomaly) for corresponding metric (See later "metric")
        
    metric: {'nab', 'binary', 'average_time', 'confusion_matrix'}. 
        Default='nab'
        Affects to output (see later: Returns)
        Changepoint problem: {'nab', 'average_time'}. 
        Standart AD problem: {'binary', 'confusion_matrix'}. 
        'nab' is Numenta Anomaly Benchmark metric
        
        'average_time' is both average delay or time to failure
        depend on situatuion.
        
        'binary': FAR, MAR, F1.
        
        'confusion_matrix' standart confusion_matrix for any point.
        
    numenta_time: 'str' for pd.Tiemdelta
        Width of detection window. Default=None.
        
    portion : float, default=0.1
        The portion is needed if numenta_time = None. 
        The width of the detection window in this case is equal 
        to a portion of the width of the length of prediction divided 
        by the number of real CPs in this dataset. Default=0.1.
        
    anomaly_window_destenation: {'lefter', 'righter', 'center'}. Defualt='right'
        The parameter of the location of the detection window relative to the anomaly. 
        'lefter'  : the detection window will be on the left side of the anomaly
        'righter' : the detection window will be on the right side of the anomaly
        'center'  : the scoring window will be positioned relative to the center of anom.
                  
    clear_anomalies_mode : boolean, default=True.
        True : then the `left value of a Scoring function is Atp and the 
        `right is Afp. Only the `first value inside the detection window is taken.
        False: then the `right value of a Scoring function is Atp and the 
        `left is Afp. Only the `last value inside the detection window is taken.

    intersection_mode: {'cut left widnow', 'cut right window', 'both'}. 
        Defualt='cut right window'
        The parameter will be used if the detection windows overlap for 
        true changepoints, which is generally undesirable and requires a 
        different approach than simply cropping the scoring window using 
        this parameter.
        'cut left window' : will cut the overlapping part of the left window
        'cut right window': will cut the intersecting part of the right window
        'both'            : will crop the intersecting portion of both the left 
        and right windows
    
    verbose:  booleant, default=True.
        If True, then output useful information
        
    plot_figure : booleant, default=False.
        If True, then drawing the score fuctions, detection windows and predictions
        It is used for example, for calibration the scale_koef. 

    table_of_coef (metric='nab'): pd.DataFrame of specific form. See bellow. 
        Application profiles of NAB metric.If Defaut is None:  
        table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                      [1.0,-0.22,1.0,-1.0],
                                      [1.0,-0.11,1.0,-2.0]])
        table_of_coef.index = ['Standart','LowFP','LowFN']
        table_of_coef.index.name = "Metric"
        table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']
        
    scale_func (metric='nab'): "default" of "improved". Defualt="improved".
        Scoring function in NAB metric.
        'default'  : standart NAB scoring function 
        'improved' : Our function for resolving disadvantages
        of standart NAB scoring function 
                  
    scale_koef : float > 0. Defualt=1.0. 
        Smoothing factor. The smaller it is, 
        the smoother the scoring function is.
                  
    Returns
    ----------
    metrics : value of metrics, depend on metric 
        'nab': tuple
            - Standart profile, float
            - Low FP profile, float
            - Low FN profilet
        'average_time': tuple
            - Average time (average delay, or time to failure)
            - Missing changepoints, int
            - FPs, int
            - Number of true changepoints, int
        'binary': tuple
            - F1 metric, float
            - False alarm rate, %, float
            - Missing Alarm Rate, %, float
        'binary': tuple
            - TPs, int
            - TNs, int
            - FPs, int
            - FNS, int 
                        
    """
    # ююююююююююююююююююююююююююююююююююююююююююююююююю
    ## проверки 
    
    assert isinstance(true, pd.Series) or isinstance(true, list)
    # проверки prediction
    if isinstance(prediction, pd.Series):
        true = [true]
        prediction = [prediction]
    elif isinstance(prediction,list):
        if  not all(isinstance(my_el, pd.Series) for my_el in prediction):
            raise Exception('Uncorrect format for prediction')
    else:
        raise Exception('Uncorrect format for prediction')
        
    # проверки длин датасетов: Number of dataset unequal
    assert len(true) == len(prediction)
    
   
    
    # проверки true
    input_variant= check_errors(true)
    
    
    def check_sort(my_list,input_variant):
        for dataset in my_list:
            if input_variant==2:
                assert all(np.sort(dataset)==np.array(dataset))
            elif input_variant==3:
                assert all(np.sort(np.concatenate(dataset))==np.concatenate(dataset))
            elif input_variant==1:
                assert all(dataset.index.values == dataset.sort_index().index.values)
    check_sort(true,input_variant)
    check_sort(prediction,1)
    
    # ююююююююююююююююююююююююююююююююююююююююююююююююю
    # part 2. To detected boundaries 
    
    if (portion is None) and (numenta_time is None) and (input_variant!=3):
        portion= 0.1
        print( f'Since you not choose numenta_time and portion, then portion will be {portion}')      
        
    
   
    if input_variant==1:
        detecting_boundaries = [single_detecting_boundaries(true_series = true[i], 
                                                            true_list_ts = None,
                                                            prediction = prediction[i],
                                                            numenta_time=numenta_time,
                                                            portion=portion,
                                                            anomaly_window_destenation=anomaly_window_destenation,
                                                            intersection_mode=intersection_mode)
                                for i in range(len(true))]
          
    elif input_variant==2:
        detecting_boundaries = [single_detecting_boundaries(true_series = None,
                                                            true_list_ts=true[i], 
                                                            prediction=prediction[i], 
                                                            numenta_time=numenta_time,
                                                            portion=portion,
                                                            anomaly_window_destenation=anomaly_window_destenation,
                                                            intersection_mode=intersection_mode)
                                    for i in range(len(true))]
          
    elif input_variant==3:
        detecting_boundaries = true.copy()
        # NExt anti fool system [[[t1,t2]],[]] -> [[[t1,t2]],[[]]] 
        for i in range(len(detecting_boundaries)): 
            if len(detecting_boundaries[i])==0:
                    detecting_boundaries[i]=[[]]   
    else:
        raise Exception('Unknown format for true')
        
        


    
    # ююююююююююююююююююююююююююююююююююююююююююююююююю
    # part 3. To compute metric
    
    #print(detecting_boundaries)
    if plot_figure:
        num_datasets = len(true)
        if  ((metric=='binary') or (metric=='confusion_matrix')) \
            and (input_variant==1):
            f = plt.figure(figsize=(16,5*num_datasets))
            grid = gridspec.GridSpec(num_datasets, 1)
            for i in range(num_datasets):
                globals()['ax'+str(i)] = f.add_subplot(grid[i])
                prediction[i].plot(ax=globals()['ax'+str(i)],label='pred',marker='o')
                true[i].plot(ax=globals()['ax'+str(i)],label='true',marker='o')
                globals()['ax'+str(i)].legend()
            plt.show()
        else:
            from .univariate_funcs import my_scale
            f = plt.figure(figsize=(16,5*num_datasets))
            grid = gridspec.GridSpec(num_datasets, 1)
            detalization = 100
            for i in range(num_datasets):
                globals()['ax'+str(i)] = f.add_subplot(grid[i])
                print_legend_boundary=True
                def plot_cp(couple,anomaly_window_destenation,ax,label):
                    if anomaly_window_destenation=='lefter':
                        ax.axvline(couple[1],c='r',label=label)
                    elif anomaly_window_destenation=='righter':
                        ax.axvline(couple[0],c='r',label=label)
                    elif anomaly_window_destenation=='center':
                        ax.axvline(couple[0]+((couple[1]-couple[0])/2),c='r',label=label)
                
                for couple in detecting_boundaries[i]:
                    if len(couple)>0:
                        globals()['ax'+str(i)].axvspan(couple[0],couple[1], alpha=0.5, color='green',
                        label='detection \nboundary' if print_legend_boundary else None)
                        nab = pd.Series(my_scale(plot_figure=True,detalization=detalization),
                                        index=pd.date_range(couple[0],couple[1],periods=detalization))
                        nab.plot(ax=globals()['ax'+str(i)], linewidth=0.4, color='brown',
                        label='nab scoring func' if print_legend_boundary else None)
                        plot_cp(couple,anomaly_window_destenation,globals()['ax'+str(i)],
                                label='Changepoint' if print_legend_boundary else None)
                        print_legend_boundary = False                                                
                    else: 
                        pass 
                prediction[i].plot(ax=globals()['ax'+str(i)],label='pred', marker='o')
                globals()['ax'+str(i)].legend()
            plt.show()
        
            
            
    
    
    if  metric=='nab':
        matrix = np.zeros((3,3))
        for i in range(len(prediction)):
            matrix_ = single_evaluate_nab(detecting_boundaries[i],
                                          prediction[i],
                                          table_of_coef=table_of_coef,
                                          clear_anomalies_mode = clear_anomalies_mode,
                                          scale_func = scale_func,
                                          scale_koef=scale_koef,
                                          plot_figure=plot_figure)
            matrix = matrix + matrix_      
                    
        results = {}
        desc = ['Standart', 'LowFP', 'LowFN'] 
        for t, profile_name in enumerate(desc):
            results[profile_name] = round(100*(matrix[0,t]-matrix[1,t])/(matrix[2,t]-matrix[1,t]), 2)
            if verbose:
                print(profile_name,' - ', results[profile_name])
        return results
    
    elif metric=='average_time':
        missing, detectHistory, FP, all_true_anom = 0, [], 0, 0
        for i in range(len(prediction)):
            missing_, detectHistory_, FP_, all_true_anom_ = single_average_delay(detecting_boundaries[i], 
                                                                                 prediction[i],
                                                                                 anomaly_window_destenation=anomaly_window_destenation,
                                                                                 clear_anomalies_mode=clear_anomalies_mode)
            missing, detectHistory, FP, all_true_anom = missing+missing_, detectHistory+detectHistory_, FP+FP_, all_true_anom+all_true_anom_
        add = np.mean(detectHistory)
        if verbose:
            print('Amount of true anomalies',all_true_anom)
            print(f'A number of missed CPs = {missing}')
            print(f'A number of FPs = {int(FP)}')
            print('Average time', add)
        return add, missing, int(FP), all_true_anom
    
    elif (metric=='binary') or (metric=='confusion_matrix') :
        
        if all(isinstance(my_el, pd.Series) for my_el in true):
            TP,TN,FP,FN = 0,0,0,0
            for i in range(len(prediction)):
                TP_,TN_,FP_,FN_ = confusion_matrix(true[i],prediction[i])
                TP,TN,FP,FN = TP+TP_,TN+TN_,FP+FP_,FN+FN_       
        else:
            print('For this metric it is better if you use pd.Series format for true \nwith common index of true and prediciton')
            TP,TN,FP,FN = 0,0,0,0
            for i in range(len(prediction)):
                dict_cp_confusion = extract_cp_confusion_matrix(detecting_boundaries[i],prediction[i],binary=True)
                TP+=np.sum([len(dict_cp_confusion['TPs'][window][1]) for window in dict_cp_confusion['TPs']])
                FP+=len(dict_cp_confusion['FPs'])
                FN+=len(dict_cp_confusion['FNs'])
                TN+= len(prediction[i]) - TP - FP - FN
            
            
        if metric=='binary':
            f1 = round(TP/(TP+(FN+FP)/2), 2)
            far = round(FP/(FP+TN)*100,2)
            mar = round(FN/(FN+TP)*100,2)
            if verbose:
                print(f'False Alarm Rate {far} %' )
                print(f'Missing Alarm Rate {mar} %')
                print(f'F1 metric {f1}')
            return f1,far,mar
        
        elif metric=='confusion_matrix':
            if verbose:
                print('TP',TP)
                print('TN',TN)
                print('FP',FP)
                print('FN',FN)
            return TP,TN,FP,FN 
    else:
        raise Exception("Choose the perfomance metric")
