import pandas as pd

from ..base.task import Task, TaskResult
from .eda import TimeDiscretizationResult, FindNaNResult, EquipmentDowntimeResult


class ScalingTaskResult(TaskResult):
    """
    Это не реальная таска, а Олега изобретение чтобы тестить пайплайн
    """

    def show(self) -> None:
        pass


class ScalingTask(Task):
    """
    Это не реальная таска, а Олега изобретение чтобы тестить пайплайн
    """

    def __init__(self, name: str | None = None):

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        super().__init__(name)

    def fit(self, df: pd.DataFrame, nan_result: FindNaNResult, downtime_result: EquipmentDowntimeResult) -> tuple[pd.DataFrame, TaskResult]:
        
        data_proc = df.copy().resample(self.descretization_result.FREQ_TOBE).mean()
        df = df.copy()
        for col in data_proc:
            if col in data_proc.columns:
                data_proc[col] = data_proc[col].fillna(method='ffill',limit=nan_result.ffill_limits[col])

        for i in range(len(downtime_result.equipment_downtimes)):
            data_proc[col].drop(data_proc[
                    downtime_result.equipment_downtimes.iloc[0]['t1']:downtime_result.equipment_downtimes.iloc[0]['t2']
                ].index)    
        
        return self.scaler.fit_transform(data_proc), ScalingTaskResult()
    
    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:

        data_proc = df.copy().resample(self.descretization_result.FREQ_TOBE).mean()
        data_proc = self.scaler.transform(data_proc)

        return data_proc, ScalingTaskResult()


class ValueRangeProcessingResult(TaskResult):

    min_values : pd.Series
    max_values : pd.Series

    def show(self) -> None:
        pass


class ValueRangeProcessingTask(Task):

    def __init__(self, 
                 name: str | None = None, 
                 method: str | None = 'auto',
                 show:  bool | None = True):
        super().__init__(name)
        
        self.show = show
        self.method = method
        
        
    def _check_intervals(self, df: pd.DataFrame, result: ValueRangeProcessingResult):
        import numpy as np
        
        mask_fault_mode = (df<result.min_values) | (df>result.max_values)
        if mask_fault_mode.sum().sum()!=0:
            import warnings
            if result.show:
                warnings.warn("Некоторые значения вышли за допустимый диапазон:")
                which_columns = mask_fault_mode.sum()!=0
                which_columns = which_columns[which_columns].index
                for col in which_columns:
                    display(df[col][mask_fault_mode[col]])
            else:
                warnings.warn("Некоторые значения вышли за допустимый диапазон. Сделай show=True, чтобы посмотреть подробнее""")
        if result.show:
            print('Значения вышедшие за интервал будут заменены на нан')
        
        df[mask_fault_mode] = np.nan
        return df
        
        
    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:

        result = ValueRangeProcessingResult()
        method = self.method
        
        if method=='auto':
            method = 'MinMax'
            
        if method == 'MinMax':
            result.min_values = df.min()
            result.max_values = df.max()    
        # if method =='Межкартильных размах'
            # mask = ((new_df < new_df.quantile(0.4)) | (new_df > new_df.quantile(0.8)))
            # df = data
            # min_values = df.median() + (df.max() - df.median())*2
            # max_values = df.median() - (df.median() - df.min())*2        
        df = self._check_intervals(df, result)        
        return df, result


    def predict(self, df: pd.DataFrame, vrp_result: ValueRangeProcessingResult) -> tuple[pd.DataFrame, TaskResult]:
        df = self._check_intervals(df, vrp_result) 
        return df, vrp_result



class ResampleProcessingResult(TaskResult):

    FREQ_TOBE : str # может наверное дельта Т быть
    def show(self) -> None:

        pass


class ResampleProcessingTask(Task):
    """
    Класс задачи предварительной обработки данных в области преобразования частоты диксретизации. 
    """
    def __init__(self, 
                 name: str | None = None, 
                 FREQ_TOBE: str | None = None,
                 ):
        """ Это задача, представляющая из себя класс, для преобразования
        частоты дискритизации временного индекса с сохранением некоторой 
        метаинформации в ResampleProcessingResult. 

        Parameters
        ----------
        FREQ_TOBE : pd.Timedelta | str, default None
            Задаваемая пользователем частота дискретизации. Если None, 
            то частота берется из результатов задачи TimeDiscretizationResult.
        """
        super().__init__(name)
        self.FREQ_TOBE = FREQ_TOBE
        
        
    def fit(self, df: pd.DataFrame, time_result:TimeDiscretizationResult) -> tuple[pd.DataFrame, ResampleProcessingResult]:
        """
        Fit the ResampleProcessingTask.
        Происходит преобразование частоты дискретизации подаваемого на вход df 
        в соответствии либо с FREQ_TOBE при инициализаии (приоритет), либо в соответствии 
        с найденной в задаче  TimeDiscretizationTask

        Parameters
        ----------

        df : pd.DataFrame 
            Входной датасет

        time_result : TimeDiscretizationResult 
            Результаты работы задачи TimeDiscretizationTask

        Returns
        -------
        new_df : pd.DataFrame 
            Выходной датасет
        
        result : TimeDiscretizationResult
            Объект сохраненных результатов задачи ResampleProcessingResult
            
        Notes
        -----
        При вызове метода fit происходит сохранение следующей информации в 
        TimeDiscretizationResult:  
            FREQ_TOBE : назначенная частота дискретизации 
        """        
        result = ResampleProcessingResult()
        print(self.FREQ_TOBE )
        result.FREQ_TOBE = self.FREQ_TOBE if self.FREQ_TOBE is not None else time_result.FREQ_TOBE
        print(result.FREQ_TOBE)
        df = df.resample(result.FREQ_TOBE).mean()
        return df, result

    def predict(self, df: pd.DataFrame, result: ResampleProcessingResult) -> tuple[pd.DataFrame, ResampleProcessingResult]:
        """
        Predict by ResampleProcessingTask.
        Происходит преобразование частоты дискретизации подаваемого на вход df 
        в соответствии с заданной на этапе fit.   

        Parameters
        ----------

        df : pd.DataFrame 
            Входной датасет

        time_result : TimeDiscretizationResult 
            Результаты работы задачи TimeDiscretizationTask

        Returns
        -------
        new_df : pd.DataFrame 
            Выходной датасет
        
        result : ResampleProcessingResult
            Объект сохраненных результатов задачи ResampleProcessingResult
            
        Notes
        -----
        При вызове метода fit происходит сохранение следующей информации в 
        TimeDiscretizationResult:  
            FREQ_TOBE : назначенная частота дискретизации 
        """    
        df = df.resample(result.FREQ_TOBE).mean()
        return df, result



### TODO
class FeatureProcessingResult(TaskResult):

    def show(self) -> None:

        pass


class FeatureProcessingTask(Task):

    def __init__(self, 
                 name: str | None = None, 
                 ):
        super().__init__(name)        
        
    def fit(self, df: pd.DataFrame, ) -> tuple[pd.DataFrame, FeatureProcessingResult]:
        result =  FeatureProcessingResult()
        return df,result

    def predict(self, df: pd.DataFrame, result: FeatureProcessingResult) -> tuple[pd.DataFrame, FeatureProcessingResult]:
        pass




class SplitByNaNResult(TaskResult):
    """Это результаты работы задачи SplitByNaNTask:
    Ничего не сохраняется и не отображается в результате 
    выполнения этой задачи. 
    """  
    

    def show(self) -> None:

        pass


class SplitByNaNTask(Task):
    """
    Класс задачи предварительной обработки данных в части разбиения исходной выборки 
    на отдельные выборки по принципу неразрывности данных.
    """
    def __init__(self, 
                 name: str | None = None, 
                 ):
        """
        Класс задачи предварительной обработки данных в части разбиения исходной выборки 
        на отдельные выборки по принципу неразрывности данных. То есть исходный df разбивается 
        на выборки максимальной длины, а пропуски из-за которых датасет разбивается вообще удаляются, 
        то есть непрерывная часть слева и справа от пропуска есть 2 выборки, которые получились из-за
        этого пропуска. Нужен для работы с последовательностями из-за требования: по отсутствию 
        пропусков и по одинаковой частоте дискретизации. 
        """
        super().__init__(name)      
        
    def fit(self, df: pd.DataFrame,time_result:ResampleProcessingResult) -> tuple[pd.DataFrame, SplitByNaNResult]:
        """
        Fit the SplitByNaNTask. 

        Parameters
        ----------

        df : pd.DataFrame 
            Входной датасет

        time_result: ResampleProcessingResult
            Результат работы ResampleProcessingTask, главным образом из-за
            агрумента FREQ_TOBE, который нужен для разбиения

        Returns
        -------
        dfs : list[pd.DataFrame]
            Выходной датасеты удовлетворяющие условию неразрывности
        
        result : SplitByNaNResult
            Объект сохраненных результатов задачи SplitByNaNTask
            
        """

        result = SplitByNaNResult()
        from ..utils.preproc import df2dfs
        dfs = df2dfs(df,resample_freq = time_result.FREQ_TOBE)
        return dfs, result

    def predict(self, df: pd.DataFrame, result: SplitByNaNResult,time_result:ResampleProcessingResult) -> tuple[pd.DataFrame, SplitByNaNResult]:
        """
        Predict by SplitByNaNTask. 

        Parameters
        ----------

        df : pd.DataFrame 
            Входной датасет

        time_result: ResampleProcessingResult
            Результат работы ResampleProcessingTask, главным образом из-за
            агрумента FREQ_TOBE, который нужен для разбиения

        Returns
        -------
        dfs : list[pd.DataFrame]
            Выходной датасеты удовлетворяющие условию неразрывности
        
        result : SplitByNaNResult
            Объект сохраненных результатов задачи SplitByNaNTask
            
        """       
        from ..utils.preproc import df2dfs
        dfs = df2dfs(df,resample_freq = time_result.FREQ_TOBE)
        return dfs, result 



class PrepareSeqSamplesResult(TaskResult):
    """Это результаты работы задачи PrepareSeqSamplesTask:
    Ничего не сохраняется и не отображается в результате 
    выполнения этой задачи. 
    """  
    

    def show(self) -> None:

        pass


class PrepareSeqSamplesTask(Task):
    """Класс задачи предварительной обработки данных в части подготовки сэмлов 
    в виде последовательности для специализированных алгоритмов. 
    """

    def __init__(self, 
                 name: str | None = None, 
                 **kwargs,
                 ):
        """Класс задачи предварительной обработки данных в части подготовки сэмлов 
        в виде последовательности для специализированных алгоритмов. 
        """

        super().__init__(name)   
        self.kwargs=kwargs    
        
    def fit(self, dfs: pd.DataFrame | list[pd.DataFrame]) -> tuple[pd.DataFrame | list[pd.DataFrame], PrepareSeqSamplesResult]:
        """
        Fit the PrepareSeqSamplesTask. 

        Parameters
        ----------

        dfs : pd.DataFrame | list[pd.DataFrame]
            Входной датасет/входные датасеты

        Returns
        -------
        dfs : lpd.DataFrame | list[pd.DataFrame] | list[list[pd.DataFrame]]
            Выходной датасеты последовательностей
        
        result : PrepareSeqSamplesResult
            Объект сохраненных результатов задачи PrepareSeqSamplesTask
            
        """
        result = PrepareSeqSamplesResult()
        from ..utils.trainTestSplitting import ts_train_test_split_dfs 
        dfs = ts_train_test_split_dfs(dfs,**self.kwargs)
        return dfs, result

    def predict(self, df: pd.DataFrame, result: PrepareSeqSamplesResult) -> tuple[pd.DataFrame, PrepareSeqSamplesResult]:
        """
        Predict by PrepareSeqSamplesTask. 

        Parameters
        ----------

        dfs : pd.DataFrame | list[pd.DataFrame]
            Входной датасет/входные датасеты

        result : PrepareSeqSamplesResult
            Объект сохраненных результатов задачи PrepareSeqSamplesTask

        Returns
        -------
        dfs : lpd.DataFrame | list[pd.DataFrame] | list[list[pd.DataFrame]]
            Выходной датасеты последовательностей
        
        result : PrepareSeqSamplesResult
            Объект сохраненных результатов задачи PrepareSeqSamplesTask
            
        """
        from ..utils.trainTestSplitting import ts_train_test_split_dfs 
        dfs = ts_train_test_split_dfs(dfs,**self.kwargs)
        return dfs, result