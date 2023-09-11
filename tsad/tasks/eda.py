import pandas as pd
import numpy as np

from ..base.task import Task, TaskResult


class HighLevelDatasetAnalysisResult(TaskResult):
    """Это результаты работы задачи HighLevelDatasetAnalysisTask:  
    
    Attributes
    ----------
    start_time : pd.DatetimeIndex
        Первый временной индекс в исходном датасете

    end_time : pd.DatetimeIndex
        Последний временной индекс в исходном датасете. 

    duration : pd.Timedelta
        Временной охват в исходном датасете.
    
    length : int
        Количество сэмплов в исходном датасете.

    columns_num : int
        Количесто колонок в исходном датасете.

    columns : list[str]
        Список колонок в исходном датасете.

    types : pd.Series
        Таблица типов данных колонок

    """


    start_time: pd.DatetimeIndex
    end_time: pd.DatetimeIndex
    duration = None
    length = None
    columns = None
    types: pd.Series

    def show(self) -> None:

        from IPython.display import display

        display(f"Датасет размером {self.length}, признаков: {self.columns_num}")
        display(f"В период с {self.start_time} по {self.end_time}")
        display(f"Общей длительностью {self.duration}")
        
        display(self.types.value_counts())
        display(self.types.sort_values())


class HighLevelDatasetAnalysisTask(Task):
    """
    Класс задачи разведочного анализа данных в области оценки общей информации о датасете. 
    """
    
    def __init__(self, name: str | None = None):
        """ Класс задачи разведочного анализа данных в области оценки общей информации о датасете. 
        Осуществляет анализ, вывод и сохранение верхнеуровневой информации о датасете.
        Сохранение осуществляется через HighLevelDatasetAnalysisResult для 
        использование полученной информации в последующих задачах, востребованных в рамках 
        верхнеуровневого пайплайна.
        
        Notes
        -----
        При вызове метода fit происходит сохранение следующей информации в 
        HighLevelDatasetAnalysisResult:  
            start_time : Первый временной индекс в исходном датасете
            end_time : Последний временной индекс в исходном датасете. 
            duration : Временной охват в исходном датасете.
            length : Количество сэмплов в исходном датасете.
            columns_num : Количесто колонок в исходном датасете.
            columns : Список колонок в исходном датасете.
            types : Таблица типов данных колонок
        """
        
        super().__init__(name)

 

    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, HighLevelDatasetAnalysisResult]:
        """
        Fit the HighLevelDatasetAnalysisTask. 

        Notes
        -----        
        В данном случае метод производит 
        сохранение информации в HighLevelDatasetAnalysisResult следующей информации: 
            start_time : Первый временной индекс в исходном датасете
            end_time : Последний временной индекс в исходном датасете. 
            duration : Временной охват в исходном датасете.
            length : Количество сэмплов в исходном датасете.
            columns_num : Количесто колонок в исходном датасете.
            columns : Список колонок в исходном датасете.
            types : Таблица типов данных колонок

        """

        from pandas.api.types import is_datetime64_any_dtype

        assert is_datetime64_any_dtype(df.index)
        assert (df.index ==  sorted(df.index)).all
        assert len(df) > 1

        result = HighLevelDatasetAnalysisResult()
        result.start_time = df.index[0]
        result.end_time = df.index[-1]
        result.duration = df.index[-1] - df.index[0]
        result.length = df.shape[0]
        result.columns_num = df.shape[1]
        result.columns = df.columns
        result.types = df.dtypes.sort_index()

        return df, result
    
    def predict(self, df: pd.DataFrame, result: HighLevelDatasetAnalysisResult) -> tuple[pd.DataFrame, HighLevelDatasetAnalysisResult]:
        """
        Predict the HighLevelDatasetAnalysisTask. 
        В данном методе ничего не происходит. Нужен для реализации верхнеуровневых пайплайнов. 
        """
        return df, result


class TimeDiscretizationResult(TaskResult):
    """Это результаты работы задачи TimeDiscretizationTask:  
    
    Attributes
    ----------
    FREQ_TOBE : pd.Timedelta | str
        Вычисленная частота дискретизации,
        которая оптимальна для данного датасета. Способ 
        получения см. подробнее в TimeDiscretizationTask. 

    frequence_of_diff_interval: pd.Series
        Таблица, где индекс - это диапазон возможных периодов
        между сэмплами, а колонка - это количество случаев в
        датасете с таким диапазоном периодов. 

    index_freq_climed: pd.Timedelta | str | None
        Единый обозначенный период исходного датасета, 
        если такой имеется. 

    min_diff : pd.Timedelta | str
        Минимальный период между сэмплами в датасете
        
    max_diff : pd.Timedelta | str
        Максимальный период между сэмплами в датасете
    
    most_frequent_diff_value : pd.Timedelta | str
        Наиболее часто встречающйся период

    most_frequent_diff_amount_cases : int
        Количество случаев с наиболее часто встречающимся периодом
               
    most_frequent_diff_amount_cases_percent : float
        Доля случаев с наиболее часто встречающимся периодом

    amount_unique_diff : int
        Количество уникальных периодов
    
    amount_unique_diff_percent : float
        Доля уникальных периодов среди общего количества сэмплов

    """

    index_freq_climed = None
    most_frequent_diff_value = None
    most_frequent_diff_amount_cases: int
    most_frequent_diff_amount_cases_percent: float
    amount_unique_diff: int
    amount_unique_diff_percent: float
    min_diff = None
    max_diff = None
    frequence_of_diff_interval: pd.Series
    FREQ_TOBE: str

    dataset_analysis_result: HighLevelDatasetAnalysisResult

    def show(self) -> None:
            
        from IPython.display import display

        display(f"В период с {self.dataset_analysis_result.start_time} по {self.dataset_analysis_result.end_time}")
        display(f"Общей длительностью {self.dataset_analysis_result.duration}")
        
        display(f"Распередление периодов между точками")
        display(self.frequence_of_diff_interval)
        
        display(f"Заявленный период {self.index_freq_climed}")
        display(f"Наиболее часто встречающйся период {self.most_frequent_diff_value}")
        display(f"Количество и доля наиболее часто встречающйся периодов {self.most_frequent_diff_amount_cases} \
                шт, это {self.most_frequent_diff_amount_cases_percent} %")
        display(f"Количество уникальных периодов {self.amount_unique_diff} при {self.dataset_analysis_result.length}\
                точек датасата, что есть {self.amount_unique_diff_percent}%")
        display(f"Минимальный период: {self.min_diff}, Максимальный период: {self.max_diff}")
        display(f"ВЫБРАН ПЕРИОД для РЕСЕМПЛИРОВНИЯ: {self.FREQ_TOBE}")


class TimeDiscretizationTask(Task):
    """
    Класс задачи разведочного анализа данных в области оценки частоты дискретизации 
    и поиска оптимальной частоты. 
    """


    def __init__(self, name: str | None = None, FREQ_TOBE=None, freq_tobe_approach: str = 'auto'):
        """ Это задача, представляющая из себя класс, для анализа и вывода иннформации 
        о частоте дискритизации временного индекса и сохранения этой информации
        в TimeDiscretizationResult. 

        Parameters
        ----------
        FREQ_TOBE : pd.Timedelta | str, default None
            Задаваемая пользователем частота дискретизации. Если не None, 
            то поиск оптимальной частоты не производится, а параметр 
            freq_tobe_approach становится равным "custom".

        freq_tobe_approach : str, default 'auto'
            Способ формирования оптимальной частоты дискретизации, которая 
            может быть востребована при дальнейших обработках. 
            Варианты: 'custom', 'min_period', 'most_frequent', 'auto'
                * Если 'custom' для оптимальной частоты берется частота из параметра FREQ_TOBE
                * Если 'min_period' для оптимальной частоты берется частота,
                соответствующая минимальному периоду между сэмплами
                * Если 'min_period' для оптимальной частоты берется частота,
                соответствующая наиболее часто встречающемуся периоду между сэмплами
                * Если 'auto' для оптимальной частоты берется частота,
                которая находится сложным образом на основе округления большего числа периодов.
                Подробнее смотри код.
            
        Notes
        -----
        При вызове метода fit происходит сохранение следующей информации в 
        TimeDiscretizationResult:  
            FREQ_TOBE : вычисленная частота дискретизации 
            frequence_of_diff_interval: распределение периодов между сэмлами 
            index_freq_climed: Единый обозначенный период исходного датасета
            min_diff : Минимальный период между сэмплами в датасете                
            max_diff : Максимальный период между сэмплами в датасете          
            most_frequent_diff_value : Наиболее часто встречающйся период
            most_frequent_diff_amount_cases : Количество случаев с наиболее часто 
            встречающимся периодом                    
            most_frequent_diff_amount_cases_percent : Доля случаев с наиболее 
            часто встречающимся периодом
            amount_unique_diff : Количество уникальных периодов
            amount_unique_diff_percent : Доля уникальных периодов среди общего количества сэмплов
            """


        super().__init__(name)
        self.freq_tobe_approach = freq_tobe_approach
        self.FREQ_TOBE = FREQ_TOBE



    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TimeDiscretizationResult]:

        """
        Fit the TimeDiscretizationTask. 

        Parameters
        ----------

        df : pd.DataFrame 
            Входной датасет

        Returns
        -------
        new_df : pd.DataFrame 
            Выходной датасет
        
        result : TimeDiscretizationResult
            Объект сохраненных результатов задачи TimeDiscretizationTask
            
        Notes
        -----
        При вызове метода fit происходит сохранение следующей информации в 
        TimeDiscretizationResult:  
            FREQ_TOBE : вычисленная частота дискретизации 
            frequence_of_diff_interval: распределение периодов между сэмлами 
            index_freq_climed: Единый обозначенный период исходного датасета
            min_diff : Минимальный период между сэмплами в датасете                
            max_diff : Максимальный период между сэмплами в датасете          
            most_frequent_diff_value : Наиболее часто встречающйся период
            most_frequent_diff_amount_cases : Количество случаев с наиболее часто 
            встречающимся периодом                    
            most_frequent_diff_amount_cases_percent : Доля случаев с наиболее 
            часто встречающимся периодом
            amount_unique_diff : Количество уникальных периодов
            amount_unique_diff_percent : Доля уникальных периодов среди общего количества сэмплов

        """


        from ..utils.preproc import value_counts_interval
        result = TimeDiscretizationResult()


        index = df.index.to_series()
        diff = index.diff()
        frequence_of_diff = diff.value_counts()

        
        result.index_freq_climed = index.index.freq

        ### считаем разницу между сэмлами
        result.most_frequent_diff_value = frequence_of_diff.index[0]
        result.most_frequent_diff_amount_cases =  frequence_of_diff.iloc[0]
        result.most_frequent_diff_amount_cases_percent = round(result.most_frequent_diff_amount_cases/len(index) *100,3)
        result.amount_unique_diff = len(frequence_of_diff)
        result.amount_unique_diff_percent = round(len(frequence_of_diff)/len(index) *100,3)

        result.min_diff = frequence_of_diff.sort_index().index[0]
        result.max_diff = frequence_of_diff.sort_index().index[-1]

        intervals=[
            pd.Timedelta('0ns'),
            pd.Timedelta('1ns'),
            pd.Timedelta('1s'),
            pd.Timedelta('1m'),
            pd.Timedelta('1h'),
            pd.Timedelta('8h'),
            pd.Timedelta('1D'),
            pd.Timedelta('7D'),
            pd.Timedelta('30D')
        ]
        result.frequence_of_diff_interval = value_counts_interval(diff,intervals)

        if self.freq_tobe_approach=='auto':
            if len(frequence_of_diff)>2:
                v1 = diff.quantile(0.05)
                v2 = diff.quantile(0.5)
                d = v2 - v1
                canbe = 'ns','us','ms','10ms','100ms','s','T', 'H','D','30D','90D','365D'
                success = False
                for freq in canbe:
                    if d != d.round(freq):
                        success = True
                        break
                if not success:
                    raise print('Не получилось найти универсальный период')
                result.FREQ_TOBE = freq
            else:
                result.FREQ_TOBE = result.most_frequent_diff_value
            
        elif self.freq_tobe_approach=='most_frequent':
            result.FREQ_TOBE = result.most_frequent_diff_value

        elif self.freq_tobe_approach=='min_period':
            result.FREQ_TOBE = result.min_diff 

        elif self.freq_tobe_approach=='custom':
            result.FREQ_TOBE = self.FREQ_TOBE
        else:
            raise Exception("Неправильный аргумент для параметра freq_tobe_approach ")

        return df, result
    
    def predict(self, df: pd.DataFrame, result: TimeDiscretizationResult ) -> tuple[pd.DataFrame, TimeDiscretizationResult]:
        """
        Predict by TimeDiscretizationTask. 
        В данном методе ничего не происходит. Нужен для реализации верхнеуровневых пайплайнов. 

        Parameters
        ----------

        df : pd.DataFrame 
            Входной датасет

        result : TimeDiscretizationResult
            Объект сохраненных результатов задачи TimeDiscretizationTask

        Returns
        -------
        new_df : pd.DataFrame 
            Выходной датасет
        
        result : TimeDiscretizationResult
            Объект сохраненных результатов задачи TimeDiscretizationTask

        """

        return df, result


class FindNaNResult(TaskResult):
    """Это результаты работы задачи FindNaNTask
    
    Attributes
    ----------

    mask_nan : pd.DataFrame
        Маска пропусков на исходный датасет

    full_nan_col_names : list[str]
        Список колонок, в которых только пропуски

    full_nan_col_numbers : int
        Количество колонок, в которых только пропуски

    full_nan_col_percent : float
        Доля колонок, в которых только пропуски

    full_nan_row_names : list
        Список строк, в которых только пропуски

    full_nan_row_numbers : int
        Количество строк, в которых только пропуски

    full_nan_row_percent : float
        Доля строк, в которых только пропуски

    total_nan_number : int
        Общее количество пропущенных ячеек таблицы

    total_nan_percent : float
        Доля пропущенных ячеек таблицы

    matrix_nan : None
        Матрица суммы попарных пересечений нанов между колонкам

    sum_nan_by_col : pd.Series
        Общее количество пропусков по колонкам

    nan_by_col : pd.DataFrame
        Таблица распределение пропусков по колонкам
    """

    mask_nan: pd.DataFrame
    full_nan_col_names: list
    full_nan_col_numbers: int
    full_nan_col_percent: float
    full_nan_row_names: list
    full_nan_row_numbers: int
    full_nan_row_percent: float
    total_nan_number = None
    total_nan_percent = None
    matrix_nan = None
    sum_nan_by_col: pd.Series
    nan_by_col: pd.DataFrame

    dataset_analysis_result: HighLevelDatasetAnalysisResult

    def show(self) -> None:
        
        from IPython.display import display
        import matplotlib.pyplot as plt

        display(f"Общее количество пропущенных ячеек таблицы  {self.total_nan_number}\
            , а это {self.total_nan_percent}%")

        
        display(f"Из  {self.dataset_analysis_result.columns_num} колонок,\
        абсолютно все пропуске в {self.full_nan_col_numbers} колонках,\
        а это {self.full_nan_col_percent}%")
        display(f" Указанные колонки это:")
        display(self.full_nan_col_names)
        
        display(f"Из ИСХОДНОГО ДАТАСАТЕ размером {self.dataset_analysis_result.length},\
        количество пропущенных строк: {self.full_nan_row_numbers},\
        а это {self.full_nan_row_percent}%")
        display(f" Указанные строки это:")
        display(self.full_nan_row_names)
        
        display(f"Распередление пропусков по колонкам")
        display(self.nan_by_col)
        
        plt.figure()
        plt.title(f"График с суммой пропусков по колонкам")
        self.sum_nan_by_col.plot()
        plt.show()
        
        display(f"Сумма совместных попарных пересечений нанов:")
        import seaborn as sns
        plt.figure()
        sns.heatmap(self.matrix_nan.astype(int), annot=True)
        plt.show()


class FindNaNTask(Task):
    """
    Класс задачи разведочного анализа данных в области оценки пропусков. 
    Рекомендуюется выполнять после очистки дубликатов и приведения к единой
    частоте дискретизации. 
    """

    def __init__(self, name: str | None = None):
        """ Класс задачи разведочного анализа данных в области оценки пропусков в датасете. 
        Осуществляет анализ, вывод и сохранение информации о пропусках в датасете.
        Сохранение осуществляется через FindNaNResult для 
        использование полученной информации в последующих задачах, востребованных в рамках 
        верхнеуровневого пайплайна.
        
        Notes
        -----
        При вызове метода fit происходит сохранение следующей информации в 
        FindNaNResult:  
            mask_nan : Маска пропусков на исходный датасет
            full_nan_col_names : Список колонок, в которых только пропуски
            full_nan_col_numbers : Количество колонок, в которых только пропуски
            full_nan_col_percent : Доля колонок, в которых только пропуски
            full_nan_row_names : Список строк, в которых только пропуски
            full_nan_row_numbers : Количество строк, в которых только пропуски
            full_nan_row_percent : Доля строк, в которых только пропуски
            total_nan_number : Общее количество пропущенных ячеек таблицы
            total_nan_percent : Доля пропущенных ячеек таблицы
            matrix_nan : Матрица суммы попарных пересечений нанов между колонкам
            sum_nan_by_col : Общее количество пропусков по колонкам
            nan_by_col : Таблица распределение пропусков по колонкам 
        """
        super().__init__(name)

    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FindNaNResult]:
        """
        Fit the FindNaNTask. 

        Parameters
        ----------

        df : pd.DataFrame 
            Входной датасет

        Returns
        -------
        new_df : pd.DataFrame 
            Выходной датасет
        
        result : FindNaNResult
            Объект сохраненных результатов задачи FindNaNResult
            
        Notes
        -----
        При вызове метода fit происходит сохранение следующей информации в 
        FindNaNResult:  
            mask_nan : Маска пропусков на исходный датасет
            full_nan_col_names : Список колонок, в которых только пропуски
            full_nan_col_numbers : Количество колонок, в которых только пропуски
            full_nan_col_percent : Доля колонок, в которых только пропуски
            full_nan_row_names : Список строк, в которых только пропуски
            full_nan_row_numbers : Количество строк, в которых только пропуски
            full_nan_row_percent : Доля строк, в которых только пропуски
            total_nan_number : Общее количество пропущенных ячеек таблицы
            total_nan_percent : Доля пропущенных ячеек таблицы
            matrix_nan : Матрица суммы попарных пересечений нанов между колонкам
            sum_nan_by_col : Общее количество пропусков по колонкам
            nan_by_col : Таблица распределение пропусков по колонкам
        """


        mask_nan = df.isin([np.inf, -np.inf, np.nan])

        full_nan_col_names = mask_nan.all(0)[mask_nan.all(0)].index.to_list()
        full_nan_col_numbers = len(full_nan_col_names)
        full_nan_col_percent = round(len(full_nan_col_names)/mask_nan.shape[1] *100,3)


        full_nan_row_names = mask_nan.all(1)[mask_nan.all(1)].index.to_list()
        full_nan_row_numbers = len(full_nan_row_names)
        full_nan_row_percent = round(len(full_nan_row_names)/mask_nan.shape[0] *100,3)

        total_nan_number = mask_nan.sum().sum()
        total_nan_percent =  round(mask_nan.sum().sum() / mask_nan.size *100,3)

        matrix_nan = mask_nan.astype(int).T @ mask_nan.astype(int)

        sum_nan_by_col = mask_nan.sum(1)

        nan_by_col = pd.Series(np.diag(matrix_nan),index=mask_nan.columns)
        nan_by_col.name = 'amount'
        if 'index' in nan_by_col.index:
            raise('index col exists, please rename this column')
        else:
            nan_by_col['index'] = mask_nan.index.isin([np.inf, -np.inf, np.nan]).sum()
        nan_by_col = nan_by_col.to_frame()
        nan_by_col['percent%']= (nan_by_col/len(mask_nan)*100).round(3)
        nan_by_col = nan_by_col.sort_values(by='percent%')

        result = FindNaNResult()
        result.mask_nan = mask_nan
        result.full_nan_col_names = full_nan_col_names
        result.full_nan_col_numbers = full_nan_col_numbers
        result.full_nan_col_percent = full_nan_col_percent
        result.full_nan_row_names = full_nan_row_names
        result.full_nan_row_numbers = full_nan_row_numbers
        result.full_nan_row_percent = full_nan_row_percent
        result.total_nan_number = total_nan_number
        result.total_nan_percent = total_nan_percent
        result.matrix_nan = matrix_nan
        result.sum_nan_by_col = sum_nan_by_col
        result.nan_by_col = nan_by_col

        return df, result
    
    def predict(self, df: pd.DataFrame,result: FindNaNResult) -> tuple[pd.DataFrame, FindNaNResult]:
        """
        Predict by FindNaNTask. 
        В данном методе ничего не происходит. Нужен для реализации верхнеуровневых пайплайнов. 

        Parameters
        ----------
        df : pd.DataFrame 
            Входной датасет

        result : FindNaNTask
            Объект сохраненных результатов задачи TimeDiscretizationTask

        Returns
        -------
        new_df : pd.DataFrame 
            Выходной датасет
        
        result : FindNaNTask
            Объект сохраненных результатов задачи TimeDiscretizationTask
        """
        return df, result


class EquipmentDowntimeResult(TaskResult):

    equipment_downtimes: pd.DataFrame

    def show(self) -> None:

        from IPython.display import display

        display(f"Все простои")
        display(self.equipment_downtimes)


class EquipmentDowntimeTask(Task):

    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, EquipmentDowntimeResult]:

        config_find_equipment_downtime = {
            'type_search': ['by_index','by_load_tag'],
            'params_local': None
        }
        type_search = 'by_index'
        ucl_delt_time = pd.Timedelta('4h')

        if type_search=='by_index':
            diff = df.dropna(how='all').index.to_series().diff()
            diff = diff[diff >=ucl_delt_time]
            diff.index.name = 't2'
            diff.name = 'duration'
            diff = diff.to_frame().reset_index()
            diff['t1'] = diff['t2'] - diff['duration'] 
            diff = diff[['t1','t2','duration']]
            equipment_downtimes = diff
        elif type_search=='by_load_tag':
            raise Exception('TODO')
        else:
            raise Exception('Нет такого аргумента')
        
        result = EquipmentDowntimeResult()
        result.equipment_downtimes = equipment_downtimes

        return df, result
    
    def predict(self, df: pd.DataFrame, result:EquipmentDowntimeResult) -> tuple[pd.DataFrame, EquipmentDowntimeResult]:
        return df, result
