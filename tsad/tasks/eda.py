import pandas as pd
import numpy as np

from ..base.task import Task, TaskResult

class HighLevelDatasetAnalysisResult(TaskResult):
    """This is the result of the HighLevelDatasetAnalysisTask:

    Attributes
    ----------
    start_time : pd.DatetimeIndex
        The first time index in the source dataset

    end_time : pd.DatetimeIndex
        The last time index in the source dataset.

    duration : pd.Timedelta
        The overall duration in the source dataset.

    length : int
        The number of samples in the source dataset.

    columns_num : int
        The number of columns in the source dataset.

    columns : list[str]
        The list of columns in the source dataset.

    types : pd.Series
        The table of data types of columns.

    """

    start_time: pd.DatetimeIndex
    end_time: pd.DatetimeIndex
    duration = None
    length = None
    columns = None
    types: pd.Series

    def show(self) -> None:
        """Displays the result of the HighLevelDatasetAnalysisTask"""

        from IPython.display import display

        display(f"Dataset size: {self.length}, features: {self.columns_num}")
        display(f"Time index from {self.start_time} to {self.end_time}")
        display(f"Total duration: {self.duration}")

        display(self.types.value_counts())
        display(self.types.sort_values())


class HighLevelDatasetAnalysisTask(Task):
    """
    Class for exploratory data analysis task to evaluate general information about the dataset.
    """

    def __init__(self, name: str | None = None):
        """Class for exploratory data analysis task to evaluate general information about the dataset.
        Performs analysis, output, and saving of high-level information about the dataset.
        Saving is done through HighLevelDatasetAnalysisResult for
        using the obtained information in subsequent tasks demanded within
        the high-level pipeline.

        Notes
        -----
        When the fit method is called, the following information is saved in
        HighLevelDatasetAnalysisResult:
            start_time : The first time index in the source dataset
            end_time : The last time index in the source dataset.
            duration : The time span in the source dataset.
            length : The number of samples in the source dataset.
            columns_num : The number of columns in the source dataset.
            columns : The list of columns in the source dataset.
            types : The table of data types of columns
        """

        super().__init__(name)

    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, HighLevelDatasetAnalysisResult]:
        """
        Fit the HighLevelDatasetAnalysisTask.
        
        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.

        Returns
        -------
        tuple[pd.DataFrame, HighLevelDatasetAnalysisResult]
            The output dataset and the result of the analysis.

        Notes
        -----
        In this case, the method saves the following information in HighLevelDatasetAnalysisResult:
            start_time : The first time index in the source dataset
            end_time : The last time index in the source dataset.
            duration : The time span in the source dataset.
            length : The number of samples in the source dataset.
            columns_num : The number of columns in the source dataset.
            columns : The list of columns in the source dataset.
            types : The table of data types of columns
        """

        start_time = df.index.min()
        end_time = df.index.max()
        duration = end_time - start_time
        length = len(df)
        columns_num = len(df.columns)
        columns = list(df.columns)
        types = df.dtypes

        result = HighLevelDatasetAnalysisResult()
        result.start_time = start_time
        result.end_time = end_time  
        result.duration = duration  
        result.length = length  
        result.columns_num = columns_num    
        result.columns = columns    
        result.types = types    


        return df, result
    
    
    def predict(self, df: pd.DataFrame, result: HighLevelDatasetAnalysisResult) -> tuple[pd.DataFrame, HighLevelDatasetAnalysisResult]:
        """
        Predict the HighLevelDatasetAnalysisTask. 
        Nothing happens in this method. Needed to implement top-level pipelines.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.
        result : HighLevelDatasetAnalysisResult
            The result of the analysis.

        Returns
        -------
        tuple[pd.DataFrame, HighLevelDatasetAnalysisResult]
            The output dataset and the result of the analysis.
        """
        return df, result
    
    

class TimeDiscretizationResult(TaskResult):
    """The result of the TimeDiscretizationTask:

    Attributes
    ----------
    freq_tobe : pd.Timedelta | str
        The computed discretization frequency that is optimal for the given dataset.
        See TimeDiscretizationTask for more details.

    frequence_of_diff_interval: pd.Series
        The table where the index is the range of possible periods between samples,
        the column is the number of cases in the dataset with such a range of periods.

    index_freq_climed: pd.Timedelta | str | None
        The unified designated period of the source dataset, if one exists.

    min_diff : pd.Timedelta | str
        The minimum period between samples in the dataset.

    max_diff : pd.Timedelta | str
        The maximum period between samples in the dataset.

    most_frequent_diff_value : pd.Timedelta | str
        The most frequently period.

    most_frequent_diff_amount_cases : int
        The number of cases with the most frequently period.

    most_frequent_diff_amount_cases_percent : float
        The proportion of cases with the most frequently occurring period.

    amount_unique_diff : int
        The number of unique periods.

    amount_unique_diff_percent : float
        The proportion of unique periods among the total number of samples.

    """
    dataset_analysis_result: HighLevelDatasetAnalysisResult
    index_freq_climed = None
    most_frequent_diff_value = None
    most_frequent_diff_amount_cases: int
    most_frequent_diff_amount_cases_percent: float
    amount_unique_diff: int
    amount_unique_diff_percent: float
    min_diff = None
    max_diff = None
    frequence_of_diff_interval: pd.Series
    freq_tobe: str

    def show(self) -> None:
        """Prints the results of the TimeDiscretizationTask."""
        from IPython.display import display
        print(dir(self))

        display(f"During the period from {self.dataset_analysis_result.start_time} to {self.dataset_analysis_result.end_time}")
        display(f"With a total duration of {self.dataset_analysis_result.duration}")
        
        display(f"Distribution of periods between points")
        display(self.frequence_of_diff_interval)
        
        display(f"Declared period {self.index_freq_climed}")
        display(f"The most frequently period {self.most_frequent_diff_value}")
        display(f"The number and proportion of the most frequently periods is {self.most_frequent_diff_amount_cases} \
                , which is {self.most_frequent_diff_amount_cases_percent} %")
        display(f"The number of unique periods {self.amount_unique_diff} out of {self.dataset_analysis_result.length}\
                points in the dataset, which is {self.amount_unique_diff_percent}%")
        display(f"Minimum period: {self.min_diff}, Maximum period: {self.max_diff}")
        display(f"SELECTED PERIOD for RESAMPLING: {self.freq_tobe}")


class TimeDiscretizationTask(Task):
    """
    A class of exploratory data analysis task for analyzing and printing information 
    about the frequency of time index discretization and saving this information 
    to TimeDiscretizationResult.

    Parameters
    ----------
    freq_tobe : pd.Timedelta | str, default None
        The user-defined discretization frequency. If not None, 
        the search for the optimal frequency is not performed, and the parameter 
        freq_tobe_approach becomes "custom".

    freq_tobe_approach : str, default 'auto', {'custom', 'min_period', 
        'most_frequent', 'auto'}
        The method of forming the optimal discretization frequency, which 
        may be required for further processing. 
            * If 'custom', the frequency from the freq_tobe parameter is taken as the optimal frequency.
            * If 'min_period', the frequency corresponding to the minimum period between samples is taken as the optimal frequency.
            * If 'most_frequent', the frequency corresponding to the most frequently occurring period between samples is taken as the optimal frequency.
            * If 'auto', the frequency that is found in a complex way based on rounding a larger number of periods is taken as the optimal frequency.
            See the code for more details.

    Notes
    -----
    When the fit method is called, the following information is saved in TimeDiscretizationResult:
        freq_tobe: the computed discretization frequency
        frequence_of_diff_interval: the distribution of periods between samples
        index_freq_climed: the unified designated period of the source dataset, if one exists
        min_diff: the minimum period between samples in the dataset
        max_diff: the maximum period between samples in the dataset
        most_frequent_diff_value: the most frequently occurring period
        most_frequent_diff_amount_cases: the number of cases with the most frequently occurring period
        most_frequent_diff_amount_cases_percent: the proportion of cases with the most frequently occurring period
        amount_unique_diff: the number of unique periods
        amount_unique_diff_percent: the proportion of unique periods among the total number of samples
    """


    def __init__(self, name: str | None = None, freq_tobe=None, freq_tobe_approach: str = 'auto'):
        super().__init__(name)
        self.freq_tobe_approach = freq_tobe_approach
        self.freq_tobe = freq_tobe



    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TimeDiscretizationResult]:
        """
        Fit the TimeDiscretizationTask. 

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.

        Returns
        -------
        tuple[pd.DataFrame, TimeDiscretizationResult]
            The output dataset and the result of the analysis.

        Notes
        -----
        When the fit method is called, the following information is saved in TimeDiscretizationResult: 
            freq_tobe: the computed discretization frequency
            frequence_of_diff_interval: the distribution of periods between samples
            index_freq_climed: the unified designated period of the source dataset, if one exists
            min_diff: the minimum period between samples in the dataset
            max_diff: the maximum period between samples in the dataset
            most_frequent_diff_value: the most frequently occurring period
            most_frequent_diff_amount_cases: the number of cases with the most frequently occurring period
            most_frequent_diff_amount_cases_percent: the proportion of cases with the most frequently occurring period
            amount_unique_diff: the number of unique periods
            amount_unique_diff_percent: the proportion of unique periods among the total number of samples

        """

        from ..utils.preproc import value_counts_interval
        result = TimeDiscretizationResult()

        index = df.index.to_series()
        diff = index.diff()
        frequence_of_diff = diff.value_counts()

        result.index_freq_climed = index.index.freq

        # Calculate the difference between samples
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
                    raise print('Could not find a universal period')
                result.freq_tobe = freq
            else:
                result.freq_tobe = result.most_frequent_diff_value
            
        elif self.freq_tobe_approach=='most_frequent':
            result.freq_tobe = result.most_frequent_diff_value

        elif self.freq_tobe_approach=='min_period':
            result.freq_tobe = result.min_diff 

        elif self.freq_tobe_approach=='custom':
            result.freq_tobe = self.freq_tobe
        else:
            raise Exception("Invalid argument for freq_tobe_approach parameter")

        return df, result
        
    def predict(self, df: pd.DataFrame, result: TimeDiscretizationResult ) -> tuple[pd.DataFrame, TimeDiscretizationResult]:
        """
        Predict by TimeDiscretizationTask. 
        This method does nothing. It is needed for implementing high-level pipelines.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.
        result : TimeDiscretizationResult
            The result of the analysis.

        Returns
        -------
        tuple[pd.DataFrame, TimeDiscretizationResult]
            The output dataset and the result of the analysis.

        """

        return df, result


class FindNaNResult(TaskResult):
    """The results of the FindNaNResult task.

    Attributes
    ----------
    mask_nan : pd.DataFrame
        The mask of NaN values in the original dataset.
    full_nan_col_names : list[str]
        The list of columns that contain only NaN values.
    full_nan_col_numbers : int
        The number of columns that contain only NaN values.
    full_nan_col_percent : float
        The percentage of columns that contain only NaN values.
    full_nan_row_names : list
        The list of rows that contain only NaN values.
    full_nan_row_numbers : int
        The number of rows that contain only NaN values.
    full_nan_row_percent : float
        The percentage of rows that contain only NaN values.
    total_nan_number : int
        The total number of NaN values in the dataset.
    total_nan_percent : float
        The percentage of NaN values in the dataset.
    matrix_nan : None
        The matrix of pairwise intersections of NaN values between columns.
    sum_nan_by_col : pd.Series
        The total number of NaN values per column.
    nan_by_col : pd.DataFrame
        The table with distribution of NaN values per column.


    Methods
    -------
    show() -> None
        Displays the results of the FindNaNTask task.
    """

    mask_nan: pd.DataFrame
    full_nan_col_names: list
    full_nan_col_numbers: int
    full_nan_col_percent: float
    full_nan_row_names: list
    full_nan_row_numbers: int
    full_nan_row_percent: float
    total_nan_number: int
    total_nan_percent: float
    matrix_nan: None
    sum_nan_by_col: pd.Series
    nan_by_col: pd.DataFrame
    dataset_analysis_result: HighLevelDatasetAnalysisResult

    def show(self) -> None:
        """Displays the results of the FindNaNTask task."""
        
        from IPython.display import display
        import matplotlib.pyplot as plt

        display(f"The total number of NaN values in the dataset is {self.total_nan_number},\
            which is {self.total_nan_percent}% of the dataset.")

        
        display(f"Out of {self.dataset_analysis_result.columns_num} columns,\
        all values are NaN in {self.full_nan_col_numbers} columns,\
        which is {self.full_nan_col_percent}%.")
        display(f"These columns are:")
        display(self.full_nan_col_names)
        
        display(f"Out of the ORIGINAL DATASET with {self.dataset_analysis_result.length} rows,\
        {self.full_nan_row_numbers} rows contain only NaN values,\
        which is {self.full_nan_row_percent}%.")
        display(f"These rows are:")
        display(self.full_nan_row_names)
        
        display(f"Distribution of NaN values per column:")
        display(self.nan_by_col)
        
        plt.figure()
        plt.title(f"Graph of the sum of NaN values per column")
        self.sum_nan_by_col.plot()
        plt.show()
        
        display(f"The sum of pairwise intersections of NaN values:")
        import seaborn as sns
        plt.figure()
        sns.heatmap(self.matrix_nan.astype(int), annot=True)
        plt.show()


class FindNaNTask(Task):
    """Class of exploratory data analysis problem in the field of gap estimation. 
    It is recommended to perform this after clearing duplicates and bringing the dataset 
    to a single sampling rate. Analyzes, displays and saves information (in FindNaNResult) 
    about gaps in the dataset. 

    Notes
    -----
    When the fit method is called, the following information is stored in FindNaNResult:
        mask_nan : The mask of NaN values in the original dataset.
        full_nan_col_names : The list of columns that contain only NaN values.
        full_nan_col_numbers : The number of columns that contain only NaN values.
        full_nan_col_percent : The percentage of columns that contain only NaN values.
        full_nan_row_names : The list of rows that contain only NaN values.
        full_nan_row_numbers : The number of rows that contain only NaN values.
        full_nan_row_percent : The percentage of rows that contain only NaN values.
        total_nan_number : The total number of NaN values in the dataset.
        total_nan_percent : The percentage of NaN values in the dataset.
        matrix_nan : The matrix of pairwise intersections of NaN values between columns.
        sum_nan_by_col : The total number of NaN values per column.
        nan_by_col : The table with distribution of NaN values per column.

    
    """

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FindNaNResult]:
        """
        Fit the FindNaNTask. 

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.

        Returns
        -------
        tuple[pd.DataFrame, FindNaNResult]
            The output dataset and the result of the analysis.
            
        Notes
        -----
        When the fit method is called, the following information is stored in FindNaNResult:
            mask_nan : The mask of NaN values in the original dataset.
            full_nan_col_names : The list of columns that contain only NaN values.
            full_nan_col_numbers : The number of columns that contain only NaN values.
            full_nan_col_percent : The percentage of columns that contain only NaN values.
            full_nan_row_names : The list of rows that contain only NaN values.
            full_nan_row_numbers : The number of rows that contain only NaN values.
            full_nan_row_percent : The percentage of rows that contain only NaN values.
            total_nan_number : The total number of NaN values in the dataset.
            total_nan_percent : The percentage of NaN values in the dataset.
            matrix_nan : The matrix of pairwise intersections of NaN values between columns.
            sum_nan_by_col : The total number of NaN values per column.
            nan_by_col : The table with distribution of NaN values per column.
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
        This method does nothing. It is needed for implementing high-level pipelines.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.
        result : FindNaNResult
            The result of the analysis.

        Returns
        -------
        tuple[pd.DataFrame, FindNaNResult]
            The output dataset and the result of the analysis.
        """
        return df, result


class EquipmentDowntimeResult(TaskResult):
    """
    The result of the EquipmentDowntimeTask task.

    Attributes
    ----------
    equipment_downtimes : pd.DataFrame
        The table with all equipment downtimes.

    Methods
    -------
    show() -> None
        Displays the results of the EquipmentDowntimeTask task.
    """

    equipment_downtimes: pd.DataFrame

    def show(self) -> None:
        """
        Displays the results of the EquipmentDowntimeTask task.
        """
        from IPython.display import display

        display(f"All downtimes")
        display(self.equipment_downtimes)


class EquipmentDowntimeTask(Task):
    """
    Class of exploratory data analysis problem in the field of equipment downtime estimation. 
    Analyzes, displays and saves information (in EquipmentDowntimeResult) about equipment downtimes in the dataset. 

    Notes
    -----
    When the fit method is called, the following information is stored in EquipmentDowntimeResult:
        equipment_downtimes : The table with all equipment downtimes.
    """
    def __init__(self, name: str | None = None):
        super().__init__(name)

    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, EquipmentDowntimeResult]:
        """
        Fit the EquipmentDowntimeTask. 

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.

        Returns
        -------
        tuple[pd.DataFrame, EquipmentDowntimeResult]
            The input dataset and the result of the analysis.
        """

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
            raise Exception('TODO') # TODO
        else:
            raise Exception('No such argument')
        
        result = EquipmentDowntimeResult()
        result.equipment_downtimes = equipment_downtimes

        return df, result
    
    def predict(self, df: pd.DataFrame, result:EquipmentDowntimeResult) -> tuple[pd.DataFrame, EquipmentDowntimeResult]:
        """
        Predict by EquipmentDowntimeTask. This method does nothing. It is needed for implementing high-level pipelines.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset.
        result : EquipmentDowntimeResult
            The result of the analysis.

        Returns
        -------
        tuple[pd.DataFrame, EquipmentDowntimeResult]
            The output dataset and the result of the analysis.
        """
        return df, result
