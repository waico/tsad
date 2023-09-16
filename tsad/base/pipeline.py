import inspect
import logging
import pandas as pd

from enum import Enum

from .task import Task, TaskResult
from .exceptions import ArgumentNotFoundException, UnsupportedTaskResultException


class PipelineMode(Enum):
    FIT = "FIT"
    PREDICT = "PREDICT"


class Pipeline():


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
                raise ArgumentNotFoundException(f'Unable to inject named argument {name}. Add it to fit/predict Pipeline method or set default value.')

        return arguments


    def _run(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.run_arguments = params
        task_df = df

        for task in self.tasks:

            task.mode = self.mode
            self._annotate_task_results(task)
            
            if self.mode == PipelineMode.FIT:
                parameters = self._create_method_parameters(task.fit, task_df)
                task_result = task.fit(**parameters)
            elif self.mode == PipelineMode.PREDICT:
                parameters = self._create_method_parameters(task.predict, task_df)
                task_result = task.predict(**parameters)
            else:
                raise Exception("Not supported pipeline mode.")

            if isinstance(task_result, pd.DataFrame | list):
                (result_df, result) = (task_result, None)
            elif isinstance(task_result, tuple):
                (result_df, result) = task_result
            else:
                raise UnsupportedTaskResultException(type(task_result))
            
            task_df = result_df.copy()

            if not result:
                continue

            self.results.append(result)

            if self.show:
                self._annotate_task_results(result)
                result.show()
        
        return task_df


    def fit(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.mode = PipelineMode.FIT
        return self._run(df, **params)
    

    def predict(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.mode = PipelineMode.PREDICT
        return self._run(df, **params)
