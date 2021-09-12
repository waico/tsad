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
        

def evaluating_change_point(true, prediction, metric='nab', 
                            numenta_time=None,
                            portion=0.1,
                            anomaly_window_destenation='left',
                            intersection_mode='cut left',
                            table_of_coef=None,
                            scale_func = "default",
                            scale_koef=1,
                            hidden_anomalies_mode = True,
                            plot_figure=False,
                            change_point_mode = True,
                            verbose=True
                            ):
    """
    
    true - both:
                list of pandas Series with binary int labels
                pandas Series with binary int labels
    prediction - both:
                      list of pandas Series with binary int labels
                      pandas Series with binary int labels
    metric: 'nab', 'binary' (FAR, MAR), 'average_delay'. Default='nab'
    
    verbose:  : booleant, default=True.
                  Если True, то вывод полезной информации в output
    
    <<< For 'nab', 'average_delay' metrics: ...
        portion : float, default=0.1
                Параметр нужен, если numenta_time=None
                Ширина scoring window в этом случае равна portion от 
                ширины всей выборки деленной на количество real CPs в 
                этой выборке
        anomaly_window_destenation: 'left', 'right', 'center'. Defualt='right'
                Параметр размещения scoring window относительно аномалии. 
                'left'  : scoring window будет по левую сторону от аномалии
                'right' : scoring window будет по правую сторону от аномалии
                'center': scoring window будет по размещено относительно центра
                          аномалии

        intersection_mode: 'cut left', 'cut right', 'both'. Defualt='right'
                Параметр будет задействован, в случае, если scoring windows
                пересекаются, что в общем случае нежалательно, и желательно
                требует иного подхода, чем просто обрезки scoring window
                c помощью данного параметра.
                'cut left'  : обрежет пересакающуюся часть левого окна
                'cut right' : обрежет пересакающуюся часть правого окна
                'both'  : обрежет пересакающуюся часть и левого и правого окна

    ... For 'nab', 'average_delay' metrics>>> 
    <<< For 'nab' metric:
        table_of_coef: pd.DataFrame of specific form. See bellow. 
                Application profiles of NAB metric.If Defaut is None:  
                table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                              [1.0,-0.22,1.0,-1.0],
                                              [1.0,-0.11,1.0,-2.0]])
                table_of_coef.index = ['Standart','LowFP','LowFN']
                table_of_coef.index.name = "Metric"
                table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']
            
    scale_func : "default" of "improved". Defualt="default".
                 Scoring function in NAB metric.
                 'default'  : standart NAB scoring function 
                 'improved' : Our function for resolving disadvantages
                              of standart NAB scoring function 
                  
    scale_koef : float > 0. Defualt=1.0. 
                 Коэффициент сглаживания. Чем меньше он, тем более
                 сглажена scoring function
                
    hidden_anomalies_mode : boolean, default=True.
                            True : тогда слева значение Scoring function равно Atp, а справа Afp
                            False: тогда слева значение Scoring function равно Afp, а справа Atp
    
    plot_figure : booleant, default=False.
                  Если True, то прорисовка score fuction, например, 
                  чтобы откалибровать scale_koef
    
    change_point_mode : boolean, default=True.
                        Если False, тогда берется не только первая точка прогноза в 
                        scoring window, а все. При этом взешиваются, суммируются и нормируется так,
                        что по сути можно просто смотреть на рез-тат. 
    
    
                
    """
    
    def binary(true, prediction):      
        """
        true - true binary series with 1 as anomalies
        prediction - trupredicted binary series with 1 as anomalies
        """
        def single_binary(true,prediction):
            true_ = true == 1 
            prediction_ = prediction == 1
            TP = (true_ & prediction_).sum()
            TN = (~true_ & ~prediction_).sum()
            FP = (~true_ & prediction_).sum()
            FN = (true_ & ~prediction_).sum()
            return TP,TN,FP,FN
            
        if type(true) != type(list()):
            TP,TN,FP,FN = single_binary(true,prediction)
        else:
            TP,TN,FP,FN = 0,0,0,0
            for i in range(len(true)):
                TP_,TN_,FP_,FN_ = single_binary(true[i],prediction[i])
                TP,TN,FP,FN = TP+TP_,TN+TN_,FP+FP_,FN+FN_       
    
        f1 = round(TP/(TP+(FN+FP)/2), 2)
        print(f'False Alarm Rate {round(FP/(FP+TN)*100,2)} %' )
        print(f'Missing Alarm Rate {round(FN/(FN+TP)*100,2)} %')
        print(f'F1 metric {f1}')
        return f1
    
    def average_delay(detecting_boundaries, prediction):
        """
        Возвращает add, missing, all_true_anom
        
        detecting_boundaries - где правая граница трушная аномалия, только так
        """
        
        def single_average_delay(detecting_boundaries, prediction):
            missing = 0
            detectHistory = []
            all_true_anom = 0
            _df_fill_bounds =  pd.DataFrame(np.ones((len(prediction),len(detecting_boundaries)))*np.nan,index=prediction.index)
            for i in range(len(detecting_boundaries)):
                t1 = detecting_boundaries[i][0]
                t2 = detecting_boundaries[i][1]
                _df_fill_bounds.loc[t1:t2,i]=1
                if prediction[t1:t2].sum()==0:
                    missing+=1
                else:
                    detectHistory.append(prediction[prediction ==1][t1:t2].index[0]-t1)
                all_true_anom+=1
            # FPs
                        
            
                
            ts_fp = pd.Series(np.ones(len(prediction)),index=prediction.index)
            ts_fp.loc[_df_fill_bounds.dropna(how='all').index]=0
            ts_fp = ts_fp * prediction
            FP = ts_fp.sum()
            return missing, detectHistory, FP, all_true_anom
            
        
        if type(prediction) != type(list()):
            missing, detectHistory, FP, all_true_anom = single_average_delay(detecting_boundaries, prediction)
        else:
            missing, detectHistory, FP, all_true_anom = 0, [], 0, 0
            for i in range(len(prediction)):
                missing_, detectHistory_, FP_, all_true_anom_ = single_average_delay(detecting_boundaries[i], prediction[i])
                missing, detectHistory, FP, all_true_anom = missing+missing_, detectHistory+detectHistory_, FP+FP_, all_true_anom+all_true_anom_

        add = pd.Series(detectHistory).mean()
        if verbose:
            print('Amount of true anomalies',all_true_anom)
            print(f'A number of missed CPs = {missing}')
            print(f'A number of FPs = {FP}')
            print('Average delay', add)
            
        return add, missing, int(FP), all_true_anom
    
    def evaluate_nab(detecting_boundaries, prediction, 
                     table_of_coef=None,
                     intersection_mode='cut left',
                     hidden_anomalies_mode = False,
                     scale_func = "default",
                     scale_koef=1,
                     plot_figure=True,
                     change_point_mode = False
                    ):
        """
        Scoring labeled time series by means of
        Numenta Anomaly Benchmark methodics
        Parameters
        ----------
        detecting_boundaries: list of list of two float values
            The list of lists of left and right boundary indices
            for scoring results of labeling
        prediction: pd.Series with timestamp indices, in which 1 
            is change point, and 0 in other case. 
        table_of_coef: pandas array (3x4) of float values
            Table of coefficients for NAB score function
            indeces: 'Standart','LowFP','LowFN'
            columns:'A_tp','A_fp','A_tn','A_fn'
        Returns
        -------
        Scores: numpy array, shape of 3, float
            Score for 'Standart','LowFP','LowFN' profile 
        Scores_null: numpy array, shape 3, float
            Null score for 'Standart','LowFP','LowFN' profile             
        Scores_perfect: numpy array, shape 3, float
            Perfect Score for 'Standart','LowFP','LowFN' profile  
        """
        def single_evaluate_nab(detecting_boundaries,
                                prediction, 
                                table_of_coef=None,
                                intersection_mode='cut left',
                                hidden_anomalies_mode = False,
                                scale_func = "default",
                                scale_koef=1,
                                plot_figure=True,
                                change_point_mode = False
                               ):
            """
            недостаттки scale_func default  -
            1 - зависит от относительного шага, а это значит, что если 
            слишком много точек в scoring window то перепад будет слишком
            жестким в середение. 
            2-   то самая левая точка не равно  Atp, а права не равна Afp
            (особенно если пррименять расплывающую множитель)
   
            hidden_anomalies_mode тогда слева от границы Atp срправа Afp,
            иначе fault mode, когда слева от границы Afp срправа Atp
            """
      
            def sigm_scale(len_ts, A_tp, A_fp, koef=1):
                x = np.arange(-int(len_ts/2), len_ts - int(len_ts/2))
                
                x = x if hidden_anomalies_mode else x[::-1]
                y = (A_tp-A_fp)*(1/(1+np.exp(5*x*koef))) + A_fp
                return y
            def my_scale(len_ts,A_tp,A_fp,koef=1):
                """ts - участок на котором надо жахнуть окно """
                x = np.linspace(-np.pi/2,np.pi/2,len_ts)
                x = x if hidden_anomalies_mode else x[::-1]
                # Приведение если неравномерный шаг.
                #x_new = x_old * ( np.pi / (x_old[-1]-x_old[0])) - x_old[0]*( np.pi / (x_old[-1]-x_old[0])) - np.pi/2
                y = (A_tp-A_fp)/2*-1*np.tanh(koef*x)/(np.tanh(np.pi*koef/2)) + (A_tp-A_fp)/2 + A_fp
                return y 
            
            if scale_func == "default":
                scale_func = sigm_scale
            elif scale_func == "improved":
                scale_func = my_scale
            else:
                raise("choose the scale_func")
            
            if table_of_coef is None:
                table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                     [1.0,-0.22,1.0,-1.0],
                                      [1.0,-0.11,1.0,-2.0]])
                table_of_coef.index = ['Standart','LowFP','LowFN']
                table_of_coef.index.name = "Metric"
                table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']

            detecting_boundaries = detecting_boundaries.copy()
            prediction = prediction.copy()
            _df_fill_bounds =  pd.DataFrame(np.ones((len(prediction),len(detecting_boundaries)))*np.nan,index=prediction.index)
            for i in range(len(detecting_boundaries)):
                _df_fill_bounds.loc[detecting_boundaries[i][0]:detecting_boundaries[i][1],i]=1
                if plot_figure:
                    plt.figure()
                    ts = _df_fill_bounds.iloc[:,i].dropna()
                    ts_profile = pd.Series(data=scale_func( len(ts),
                                                            table_of_coef['A_tp']['Standart'],
                                                            table_of_coef['A_fp']['Standart'],
                                                            koef=scale_koef),
                                           index = ts.index)
                    plt.plot(ts_profile,label='profile')
                    ind_ = prediction[prediction==1].loc[ts.index[0]:ts.index[-1]].index
                    if len(ind_)>0:
                        for pred_anom in ind_:
                            plt.axvline(pred_anom,c='r',label='predicted anomaly')
                        plt.legend()
                        plt.title(str(i))
                        plt.show()
                
            
            Scores, Scores_perfect, Scores_null=[], [], []
            for profile in ['Standart', 'LowFP', 'LowFN']:       
                A_tp = table_of_coef['A_tp'][profile]
                A_fp = table_of_coef['A_fp'][profile]
                A_fn = table_of_coef['A_fn'][profile]
                
                score = 0
                # FPs
                ts_fp = pd.Series(np.ones(len(prediction)),index=prediction.index)
                ts_fp.loc[_df_fill_bounds.dropna(how='all').index]=0
                ts_fp = ts_fp * prediction
                score += A_fp*ts_fp.sum()
                #FNs and TPs
                for i in range(len(detecting_boundaries)):
                    ts_tp = _df_fill_bounds.iloc[:,i].dropna()
                    ts_tp = ts_tp * prediction.loc[ts_tp.index]
                    if ts_tp.sum()==0:
                        score+=A_fn
                    else:
                        ts_profile = pd.Series(data=scale_func(
                                                                len(ts_tp),A_tp,A_fp,koef=scale_koef),
                                               index = ts_tp.index)
                        if change_point_mode:
                            ts_tp.loc[ts_tp[ts_tp==1].index[1:]] = 0
                            score += (ts_profile * ts_tp).sum()
                        else:
                            score += (ts_profile*ts_tp).sum()/(ts_profile).sum()*len(ts_profile)
                Scores.append(score)
                Scores_perfect.append(len(detecting_boundaries)*A_tp)
                Scores_null.append(len(detecting_boundaries)*A_fn)
            return np.array([np.array(Scores),np.array(Scores_null), np.array(Scores_perfect)])
       #======      
        if type(prediction) != type(list()):
            matrix = single_evaluate_nab(detecting_boundaries, 
                                         prediction, 
                                         table_of_coef=table_of_coef,
                                         intersection_mode=intersection_mode,
                                         hidden_anomalies_mode = hidden_anomalies_mode,
                                         scale_func = scale_func,
                                         scale_koef=scale_koef,
                                         plot_figure=plot_figure,
                                         change_point_mode = change_point_mode)
        else:
            matrix = np.zeros((3,3))
            for i in range(len(prediction)):
                matrix_ = single_evaluate_nab(detecting_boundaries[i],
                                              prediction[i],
                                              table_of_coef=table_of_coef,
                                              intersection_mode=intersection_mode,
                                              hidden_anomalies_mode = hidden_anomalies_mode,
                                              scale_func = scale_func,
                                              scale_koef=scale_koef,
                                              plot_figure=plot_figure,
                                              change_point_mode = change_point_mode)
                matrix = matrix + matrix_      
                
        results = {}
        desc = ['Standart', 'LowFP', 'LowFN'] 
        for t, profile_name in enumerate(desc):
            results[profile_name] = round(100*(matrix[0,t]-matrix[1,t])/(matrix[2,t]-matrix[1,t]), 2)
            print(profile_name,' - ', results[profile_name])
        
        return results
            
            
    #=========================================================================
    if type(true) != type(list()):
        true_1_indexes = true[true==1].index
    else:
        true_1_indexes = [true[i][true[i]==1].index for i in range(len(true))]
        

    if not metric=='binary':
        if metric=='average_delay':
            anomaly_window_destenation= 'left'
        def single_detecting_boundaries(true, prediction, numenta_time, true_1_indexes,
                                        anomaly_window_destenation='left',portion=0.1):
            detecting_boundaries=[]
            td = pd.Timedelta(numenta_time) if numenta_time is not None else \
                        pd.Timedelta((true.index[-1]-true.index[0])/len(true_1_indexes)*portion)  
            for val in true_1_indexes:
                if anomaly_window_destenation == 'left':
                    detecting_boundaries.append([val, val + td])
                elif anomaly_window_destenation == 'right':
                    detecting_boundaries.append([val - td, val])
                elif anomaly_destenation == 'center':
                    anomaly_window_destenation.append([val - td/2, val + td/2])
                else:
                    raise('choose anomaly_window_destenation')
                    
                    
            # block for resolving intersection problem:
            # важно не ошибиться, и всегда следить, чтобы везде правая граница далее
            # не включалась, иначе будет пересечения окон             
            new_detecting_boundaries = detecting_boundaries.copy()
            if new_detecting_boundaries[0][0] < prediction.index[0]:
                new_detecting_boundaries[0][0] = prediction.index[0]
            if new_detecting_boundaries[-1][-1] > prediction.index[-1]:
                new_detecting_boundaries[-1][-1] = prediction.index[-1]
            for i in range(len(new_detecting_boundaries)-1):
                if new_detecting_boundaries[i][1] >= new_detecting_boundaries[i+1][0]:
                    print(f'Intersection of scoring windows{new_detecting_boundaries[i][1], new_detecting_boundaries[i+1][0]}')
                    if intersection_mode == 'cut left':
                        new_detecting_boundaries[i][1] = new_detecting_boundaries[i+1][0]
                    elif intersection_mode == 'cut right':
                        new_detecting_boundaries[i+1][0] = new_detecting_boundaries[i][1]
                    elif intersection_mode == 'cut both':
                        _a  = new_detecting_boundaries[i][1]
                        new_detecting_boundaries[i][1] = new_detecting_boundaries[i+1][0]
                        new_detecting_boundaries[i+1][0] = _a
                    else:
                        raise("choose the intersection_mode")
            detecting_boundaries = new_detecting_boundaries.copy()        
                    
                    
                    
            
            return detecting_boundaries
        
        if type(true) != type(list()):
            detecting_boundaries = single_detecting_boundaries(true=true,
                                                               prediction=prediction, 
                                                               numenta_time=numenta_time, 
                                                               true_1_indexes=true_1_indexes,
                                                               anomaly_window_destenation=anomaly_window_destenation,
                                                               portion=portion
                                                              )
        else:
            detecting_boundaries=[]
            for i in range(len(true)):
                detecting_boundaries.append(single_detecting_boundaries(true=true[i], 
                                                                        prediction=prediction[i],                                                                
                                                                        numenta_time=numenta_time, 
                                                                        true_1_indexes=true_1_indexes[i],
                                                                        anomaly_window_destenation=anomaly_window_destenation,
                                                                        portion=portion))


    if metric== 'nab':
        return evaluate_nab(detecting_boundaries, 
                            prediction,
                            table_of_coef=table_of_coef,
                            intersection_mode=intersection_mode,
                            hidden_anomalies_mode = hidden_anomalies_mode,
                            scale_func = scale_func,
                            scale_koef=scale_koef,
                            plot_figure=plot_figure,
                            change_point_mode = change_point_mode)
    elif metric=='average_delay':
        return average_delay(detecting_boundaries, prediction)
    elif metric== 'binary':
        return binary(true, prediction)
        
        
def df2dfs(df,  # Авторы не рекомендуют так делать,
            resample_freq = None, # требования
            thereshold_gap = None, 
            koef_freq_of_gap = 1.2, # 1.2 проблема которая возникает которую 02.09.2021 я написал в ИИ 
            plot = True,
            col = None):
    """
    Функция которая преообратает raw df до требований к входу на DL_AD:
    
    то есть разбивает df на лист of dfs by gaps 
    
    Не ресемлирует так как это тяжелая задача, но если частота реже чем 
    koef_freq_of_gap of thereshold_gap то это воспринимается как пропуск. 
    Основной посыл: если сигнал приходит чаще, то он не уползает сильно, 
    а значит не приводит к аномалии, а если редко то приводит, поэтому воспри-
    нимается как пропуск. 
    
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
    
    