import inspect
import logging
import pandas as pd

from enum import Enum

from .task import Task, TaskResult
from .exceptions import ArgumentNotFoundException, UnsupportedTaskResultException


class PipelineMode(Enum):
    FIT_PREDICT = "FIT_PREDICT"
    PREDICT = "PREDICT"


class Pipeline():

    """
    
    ## Pipeline

    The `Pipeline` class represents a data processing pipeline that consists of multiple tasks. It allows for fitting the pipeline and predict on a training dataset and making predictions on a test dataset.

    ### Parameters

    - `tasks` (list[Task]): List of tasks to be executed in the pipeline.
    - `results` (list[TaskResult], optional): List of task results that should be stored and accessible for annotation in later tasks. Default is `None`.
    - `show` (bool, optional): Specifies whether to show the annotated task results during pipeline execution. Default is `False`.

    ### Attributes

    - `mode` (PipelineMode): The current mode of the pipeline. Can be "FIT_PREDICT" or "PREDICT".
    - `run_arguments` (dict[str, any]): The arguments passed to the `fit_predict` or `predict` method.

    ### Methods

    #### \_\_init\_\_(tasks: List[Task], results: List[TaskResult] = None, show: bool = False) -> None

    Initializes a new instance of the `Pipeline` class.

    Parameters:
    - `tasks` (list[Task]): List of tasks to be executed in the pipeline.
    - `results` (list[TaskResult], optional): List of task results that should be stored and accessible for annotation in later tasks. Default is `None`.
    - `show` (bool, optional): Specifies whether to show the annotated task results during pipeline execution. Default is `False`.

    #### \_get_result_by_type(result_type) -> TaskResult

    Returns the task result of a specified type from the `results` list.

    Parameters:
    - `result_type` (TaskResult): The type of the task result to retrieve.

    Returns:
    - `TaskResult`: The task result of the specified type.

    Raises:
    - `Exception`: If the required task result of the specified type cannot be found in the `results` list.
    - `Exception`: If multiple task results of the specified type are found in the `results` list.

    #### \_annotate_task_results(object_to_annotate) -> None

    Annotates the specified object with the task results.

    Parameters:
    - `object_to_annotate`: The object to annotate with the task results.

    #### \_create_method_parameters(method, df: pd.DataFrame) -> dict

    Creates a dictionary of method parameters for a task.

    Parameters:
    - `method`: The method for which to create the parameters.
    - `df` (pd.DataFrame): The input DataFrame for the task.

    Returns:
    - `dict`: The dictionary of method parameters.

    #### \_run(df: pd.DataFrame, \*\*params) -> pd.DataFrame

    Runs the pipeline on the specified DataFrame.

    Parameters:
    - `df` (pd.DataFrame): The input DataFrame for the pipeline.
    - `params` (keyword arguments): Additional parameters to be passed to the pipeline.

    Returns:
    - `pd.DataFrame`: The resulting DataFrame after applying all tasks in the pipeline.

    Raises:
    - `Exception`: If the pipeline mode is not supported.

    #### fit_predict(df: pd.DataFrame, \*\*params) -> pd.DataFrame

    Fits and predicts the pipeline on the specified training DataFrame.

    Parameters:
    - `df` (pd.DataFrame): The training DataFrame for fitting the pipeline and predict.
    - `params` (keyword arguments): Additional parameters to be passed to the pipeline.

    Returns:
    - `pd.DataFrame`: The resulting DataFrame after applying all tasks in the pipeline.

    #### predict(df: pd.DataFrame, \*\*params) -> pd.DataFrame

    Makes predictions using the fitted pipeline on the specified test DataFrame.

    Parameters:
    - `df` (pd.DataFrame): The test DataFrame for making predictions.
    - `params` (keyword arguments): Additional parameters to be passed to the pipeline.

    Returns:
    - `pd.DataFrame`: The resulting DataFrame of predictions.
    
    """


    mode: PipelineMode

    run_arguments: dict[str, any]


    def __init__(self, tasks: list[Task], results: list[TaskResult] = None, show: bool = False) -> None:
        self.tasks = tasks
        self.results = list()
        if results:
            self.results = results
        self.show = show


    def _get_result_by_type(self, result_type) -> TaskResult:

        results = [x for x in self.results if isinstance(x, result_type)]
        if not results:
            raise Exception(f'Can\'t find required task result of {result_type.__name__} in Pipeline.')
        if len(results) > 1:
            raise Exception('Multiple task results find!')
        return results[0]


    def _annotate_task_results(self, object_to_annotate) -> None:
        if '__annotations__' in vars(type(object_to_annotate)):
            for annotation_name, annotation_type in vars(type(object_to_annotate))['__annotations__'].items():
                if not annotation_type:
                    continue
                if issubclass(annotation_type, TaskResult):
                    result = self._get_result_by_type(annotation_type)
                    setattr(object_to_annotate, annotation_name, result)

    
    def _create_method_parameters(self, method, df: pd.DataFrame) -> dict:

        arguments = dict()

        signature = inspect.signature(method)

        (first_name, first_parameter), *parameters = signature.parameters.items()
        # TODO: Check first argument is DataFrame

        arguments[first_name] = df

        for name, parameter in parameters:

            if issubclass(parameter.annotation, TaskResult):
                arguments[name] = self._get_result_by_type(parameter.annotation)
            
            elif name in self.run_arguments:
                arguments[name] = self.run_arguments[name]
            
            elif not parameter.kind == inspect.Parameter.VAR_KEYWORD and parameter.default == signature.empty:
                raise ArgumentNotFoundException(f'Unable to inject named argument {name}. Add it to fit_predict/predict Pipeline method or set default value.')

        return arguments


    def _run(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.run_arguments = params
        task_df = df

        for task in self.tasks:

            task.mode = self.mode
            self._annotate_task_results(task)
            
            if self.mode == PipelineMode.FIT_PREDICT:
                parameters = self._create_method_parameters(task.fit_predict, task_df)
                task_result = task.fit_predict(**parameters)
            elif self.mode == PipelineMode.PREDICT:
                parameters = self._create_method_parameters(task.predict, task_df)
                task_result = task.predict(**parameters)
            else:
                raise Exception("Not supported pipeline mode.")

            if task_result is None:
                continue
            elif isinstance(task_result, pd.DataFrame | list):
                (result_df, result) = (task_result, None)
            elif isinstance(task_result, tuple):
                (result_df, result) = task_result
            else:
                raise UnsupportedTaskResultException(f'{type(task_result)} in {type(task)}')
            
            task_df = result_df.copy()

            if not result:
                continue

            if not any(result == r for r in self.results):
                self.results.append(result)

            if self.show:
                self._annotate_task_results(result)
                result.show()
        
        return task_df


    def fit_predict(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.mode = PipelineMode.FIT_PREDICT
        return self._run(df, **params)
    

    def predict(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.mode = PipelineMode.PREDICT
        return self._run(df, **params)
