import logging
import pandas as pd

from enum import Enum

from .task import Task, TaskResult


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

        parameters = dict()

        (df_name, df_type), *annotations = method.__annotations__.items()
        if not issubclass(df_type, pd.DataFrame):
            raise Exception('First argument type is not pd.DataFrame type!')

        parameters[df_name] = df

        for annotation_name, annotation_type in annotations:
            if annotation_name == 'return':
                continue
            if issubclass(annotation_type, TaskResult):
                parameters[annotation_name] = self._get_result_by_type(annotation_type)
                logging.debug(f'Adding parameter {annotation_name} with type {annotation_type.__name__} from Pipeline results.')
            
            elif annotation_name in self.run_arguments:
                parameters[annotation_name] = self.run_arguments[annotation_name]
            
            else:
                raise Exception(f'Unable to inject named argument {annotation_name}. Add it to fit/predict Pipeline method.')
        
        return parameters


    def _run(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.run_arguments = params
        task_df = df.copy()

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
            
            if task_result is None or not isinstance(task_result, tuple):
                continue

            (result_df, result) = task_result
            if self.show:
                self._annotate_task_results(result)
                result.show()

            self.results.append(result)

            task_df = result_df.copy()
        
        return task_df


    def fit(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.mode = PipelineMode.FIT
        return self._run(df, **params)
    

    def predict(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        
        self.mode = PipelineMode.PREDICT
        return self._run(df, **params)
