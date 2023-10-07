import pandas as pd

from ..base.task import Task, TaskResult
from .eda import TimeDiscretizationResult



class ValueRangeProcessingResult(TaskResult):

    min_values : pd.Series
    max_values : pd.Series

    def show(self) -> None:
        pass


class ValueRangeProcessingTask(Task):
    """
    Class of the data preprocessing task in the field of value range processing.
    """
    def __init__(self, 
                 name: str | None = None, 
                 method: str | None = 'auto',
                 show:  bool | None = True):
        """
        This is a task that represents a class for value range processing with some meta-information 
        stored in ValueRangeProcessingResult.

        Parameters
        ----------
        method : str, default 'auto'
            The method of value range processing. If 'auto', then the method is set to 'MinMax'.
        show : bool, default True
            Whether to show the warning message if some values are out of the allowed range.
        """
        super().__init__(name)
        
        self.show = show
        self.method = method
        
        
    def _check_intervals(self, df: pd.DataFrame, result: ValueRangeProcessingResult):
        """
        Check if the values in the dataframe are within the allowed range and replace the values outside the range with NaN.

        Parameters
        ----------
        df : pd.DataFrame 
            Input dataset
        result : ValueRangeProcessingResult
            Result of the ValueRangeProcessingTask

        Returns
        -------
        pd.DataFrame
            Dataframe with values outside the allowed range replaced with NaN
        """
        import numpy as np
        
        mask_fault_mode = (df<result.min_values) | (df>result.max_values)
        if mask_fault_mode.sum().sum()!=0:
            import warnings
            if result.show:
                warnings.warn("Some values are out of the allowed range:")
                which_columns = mask_fault_mode.sum()!=0
                which_columns = which_columns[which_columns].index
                for col in which_columns:
                    display(df[col][mask_fault_mode[col]])
            else:
                warnings.warn("Some values are out of the allowed range. Set show=True to see more details""")
        if result.show:
            print('Values outside the allowed range will be replaced with NaN')
        
        df[mask_fault_mode] = np.nan
        return df
        
        
    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
        """
        Fit the ValueRangeProcessingTask.
        The method finds the minimum and maximum values of the input dataframe and replaces the values outside the allowed range with NaN.

        Parameters
        ----------
        df : pd.DataFrame 
            Input dataset

        Returns
        -------
        tuple[pd.DataFrame, TaskResult]
            Dataframe with values outside the allowed range replaced with NaN and ValueRangeProcessingResult
        """
        result = ValueRangeProcessingResult()
        method = self.method
        
        if method=='auto':
            method = 'MinMax'
            
        if method == 'MinMax':
            result.min_values = df.min()
            result.max_values = df.max()    
        # if method =='Interquartile range'
            # mask = ((new_df < new_df.quantile(0.4)) | (new_df > new_df.quantile(0.8)))
            # df = data
            # min_values = df.median() + (df.max() - df.median())*2
            # max_values = df.median() - (df.median() - df.min())*2        
        df = self._check_intervals(df, result)        
        return df, result


    def predict(self, df: pd.DataFrame, vrp_result: ValueRangeProcessingResult) -> tuple[pd.DataFrame, TaskResult]:
        """
        Predict using the ValueRangeProcessingTask.
        The method replaces the values outside the allowed range with NaN.

        Parameters
        ----------
        df : pd.DataFrame 
            Input dataset
        vrp_result : ValueRangeProcessingResult
            Result of the ValueRangeProcessingTask

        Returns
        -------
        tuple[pd.DataFrame, TaskResult]
            Dataframe with values outside the allowed range replaced with NaN and ValueRangeProcessingResult
        """
        df = self._check_intervals(df, vrp_result) 
        return df, vrp_result



class ResampleProcessingResult(TaskResult):

    freq_tobe : str # can be pd.Timedelta or str
    def show(self) -> None:

        pass


class ResampleProcessingTask(Task):
    """
    Class of the data preprocessing task in the field of resampling. 
    This is a task that represents a class for resampling the frequency of the time index 
    with some meta-information stored in ResampleProcessingResult.

    Parameters
    ----------
    freq_tobe : pd.Timedelta | str, default None
        The desired frequency of the time index. If None, 
        the frequency is taken from the results of the TimeDiscretizationTask.
       
    """
    def __init__(self, 
                 name: str | None = None, 
                 freq_tobe: str | None = None,
                 ):
        super().__init__(name)
        self.freq_tobe = freq_tobe
        
        
    def fit_predict(self, df: pd.DataFrame, time_result:TimeDiscretizationResult) -> tuple[pd.DataFrame, ResampleProcessingResult]:
        """
        Fit the ResampleProcessingTask.
        The method resamples the input dataframe according to freq_tobe or the frequency found in the TimeDiscretizationTask.

        Parameters
        ----------
        df : pd.DataFrame 
            Input dataset
        time_result : TimeDiscretizationResult
            Result of the TimeDiscretizationTask

        Returns
        -------
        tuple[pd.DataFrame, ResampleProcessingResult]
            Resampled dataframe and ResampleProcessingResult
        """
        result = ResampleProcessingResult()
        if self.freq_tobe is None:
            result.freq_tobe = time_result.freq_tobe
        else:
            result.freq_tobe = self.freq_tobe
            
        df = df.resample(result.freq_tobe).mean()
        
        return df, result


    def predict(self, df: pd.DataFrame, rp_result: ResampleProcessingResult) -> tuple[pd.DataFrame, TaskResult]:
        """
        Predict using the ResampleProcessingTask.
        The method resamples the input dataframe according to the frequency found in the ResampleProcessingResult.

        Parameters
        ----------
        df : pd.DataFrame 
            Input dataset
        rp_result : ResampleProcessingResult
            Result of the ResampleProcessingTask

        Returns
        -------
        tuple[pd.DataFrame, TaskResult]
            Resampled dataframe and ResampleProcessingResult
        """
        df = df.resample(rp_result.freq_tobe).mean()
        return df, rp_result




class SplitByNaNResult(TaskResult):
    """The results of the SplitByNaNTask task: 
    Nothing is saved or displayed as a result of this task.
    """  
    

    def show(self) -> None:

        pass


class SplitByNaNTask(Task):
    """
    Class for preprocessing data by splitting the original dataset 
    into separate datasets based on the continuity of the data.

    A class of data preprocessing task in terms of dividing the original sample 
    into separate minidatasets according to the principle of data continuity. 
    That is, the original df is divided into samples of the maximum length, 
    by the gaps. These gaps,  due to which the dataset is broken, are completely 
    will be removed, that is, the continuous part to the left and to the right of 
    the gap there give us 2 minidatasets that were obtained splitting by this gap. 
    Needed for working with sequences due to the requirement: 
    the absence of gaps and the same sampling frequency.

    Parameters
    ----------
    name : str | None
        The name of the task.
    """
    def __init__(self, 
                 name: str | None = None, 
                 ):
        super().__init__(name)      
        
    def fit_predict(self, df: pd.DataFrame,time_result:ResampleProcessingResult) -> tuple[pd.DataFrame, SplitByNaNResult]:
        """
        Fits the SplitByNaNTask.

        Parameters
        ----------
        df : pd.DataFrame 
            The input dataset.

        time_result: ResampleProcessingResult
            The result of the ResampleProcessingTask, 
            mainly due to the freq_tobe argument, 
            which is needed for splitting.

        Returns
        -------
        dfs : list[pd.DataFrame]
            The output datasets that satisfy the continuity condition.
        
        result : SplitByNaNResult
            The object that stores the results of the SplitByNaNTask.
        """
        result = SplitByNaNResult()
        from ..utils.preproc import df2dfs
        dfs = df2dfs(df,resample_freq = time_result.freq_tobe)
        return dfs, result

    def predict(self, df: pd.DataFrame, result: SplitByNaNResult,time_result:ResampleProcessingResult) -> tuple[pd.DataFrame, SplitByNaNResult]:
        """
        Predicts by SplitByNaNTask.

        Parameters
        ----------
        df : pd.DataFrame 
            The input dataset.

        time_result: ResampleProcessingResult
            Need freq_tobe parametr for splitting, which is stored 
            in the ResampleProcessingTask. 

        Returns
        -------
        dfs : list[pd.DataFrame]
            The output datasets that satisfy the continuity condition.
        
        result : SplitByNaNResult
            The object that stores the results of the SplitByNaNTask.
        """       
        from ..utils.preproc import df2dfs
        dfs = df2dfs(df,resample_freq = time_result.freq_tobe)
        return dfs, result



class PrepareSeqSamplesResult(TaskResult):
    """The results of the PrepareSeqSamplesTask task:
    Nothing is saved or displayed as a result of this task.
    """  
    

    def show(self) -> None:

        pass


class PrepareSeqSamplesTask(Task):
    """
    Class of the data preprocessing task in the part of preparing samples 
    in the form of a sequence for sequence processing algorithms 
    and train test spitting execution.
    
    Parameters
    ----------
    name : (str | None) 
        Optional name for the task.

    Next parameters is used in ts_train_test_split function of utils subpackage.  
    https://tsad.readthedocs.io/en/latest/tsad.utils.html#tsad.utils.trainTestSplitting.ts_train_test_split 

    len_seq : int, default=10
        Length of the sequence, which is used to predict the next point/points.

    points_ahead : int, default=0
        How many points ahead we predict, reflected in y
        
    gap :  int, default=0
        The gap between last point of sequence, which we used as input 
        for prediction and first point of potential model output sequence
        (prediction).If the last point of input sequence is t, then the 
        first point of the output sequence is t + gap +1. The parameter 
        is designed to be able to predict sequence after a additional time 
        interval.

    step :  int, default=1.
        Sample generation step. If the first point was t for 
        the 1st sample (sequence) of the train, then for the 2nd sample 
        (sequence) of the train it will be t + step if intersection=True,
        otherwise the same but without intersections of the series values.

    intersection :  bool, default=True
        The presence of one point in time in different samples (sequences) 
        for the train set and and separately for the test test. 
        If True, the train and the test never have common time points.

    test_size : float or int or timestamp for df, or list of timestamps, default=0.25.
        The size of the test set. 
        - If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. 
        - If int, represents the absolute number of test samples. If None, the value is set to the
            complement of the train size. 
        - If 0, then it will return the X,y values in X_train, y_train. 
        - If timestamp, for X_test we will use set from df[t:] 
        - If list of timestamps [t1,t2], for X_test we will use set from df[t1:t2] 
        - If ``train_size`` is None, it will be set to 0.25. *

        
    train_size : float or int, default=None.
        The size of the train set.
        - If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. 
        - If int, represents the absolute number of train samples. 
        - If 0, then it will return the X,y values in X_test, y_test.  
        - If timestamp for df, for X_train we will use set for train from df[:t]  
        - If list of timestamps [t1,t2], for X_train we will use set for train from df[t1:t2]  
        - If None,the value is automatically set to the complement of the test size.
        
    what_to_shuffle: {'nothing', 'all','train'}, str. Default = 'train'. 
        - If 'train' we random shuffle only X_train, and y_train. 
            Test samples are unused for the shuffle. Any sample from X_test is later
            than any sample from X_train. This is also true for respectively
        - If 'all' in analogy with sklearn.model_selection.train_test_split
        - If 'nothing' shuffle is not performed.
        
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.*       
        """

    def __init__(self, 
                name: str | None = None, 
                len_seq=10, points_ahead=1, gap=0, step=1, intersection=True,
                test_size=None,train_size=None, random_state=None, what_to_shuffle='train'
                ):

        super().__init__(name)   

        self.kwargs= {}
        self.kwargs['len_seq'] = len_seq
        self.kwargs['points_ahead'] = points_ahead
        self.kwargs['gap'] = gap    
        self.kwargs['step'] = step
        self.kwargs['intersection'] = intersection
        self.kwargs['test_size'] = test_size
        self.kwargs['train_size'] = train_size
        self.kwargs['random_state'] = random_state
        self.kwargs['what_to_shuffle'] = what_to_shuffle

        
    def fit_predict(self, df: pd.DataFrame | list[pd.DataFrame]) -> tuple[pd.DataFrame | list[pd.DataFrame], PrepareSeqSamplesResult]:
        """
        Fit the PrepareSeqSamplesTask. 

        Parameters
        ----------

        dfs : pd.DataFrame | list[pd.DataFrame]
            The input dataset / input datasets

        Returns
        -------
        dfs : pd.DataFrame | list[pd.DataFrame] | list[list[pd.DataFrame]]
            The output datasets of sequences
        
        result : PrepareSeqSamplesResult
            The object that stores the results of the PrepareSeqSamplesTask task
            
        """
        result = PrepareSeqSamplesResult()
        from ..utils.trainTestSplitting import ts_train_test_split_dfs 
        df = ts_train_test_split_dfs(df,**self.kwargs)
        return df, result

    def predict(self, df: pd.DataFrame, result: PrepareSeqSamplesResult) -> tuple[pd.DataFrame, PrepareSeqSamplesResult]:
        """
        Predict by PrepareSeqSamplesTask. 

        Parameters
        ----------

        df : pd.DataFrame | list[pd.DataFrame]
            The input dataset / input datasets

        result : PrepareSeqSamplesResult
            The object that stores the results of the PrepareSeqSamplesTask task

        Returns
        -------
        df : pd.DataFrame | list[pd.DataFrame] | list[list[pd.DataFrame]]
            The output datasets of sequences
        
        result : PrepareSeqSamplesResult
            The object that stores the results of the PrepareSeqSamplesTask task
            
        """
        from ..utils.trainTestSplitting import ts_train_test_split_dfs 
        self.kwargs['test_size'] = 0 
        self.kwargs['what_to_shuffle'] = 'nothing'
        df = ts_train_test_split_dfs(df,**self.kwargs)
        return df, result