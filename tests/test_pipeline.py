import os
import sys
import pytest
import pandas as pd

from tsad.base.pipeline import Pipeline
from tsad.base.task import Task, TaskResult


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TSAD_DIR = os.path.abspath(os.path.join(TESTS_DIR, os.pardir))
sys.path.insert(0, TSAD_DIR)


class FirstTestTaskResult(TaskResult):

    length: int

    def show():
        pass


class FistTestTask(Task):


    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
        result = FirstTestTaskResult()
        result.length = len(df)
        return df, result


    def predict(self, df: pd.DataFrame, fit_result: FirstTestTaskResult) -> tuple[pd.DataFrame, TaskResult]:
        assert len(df) == fit_result.length
        return pd.DataFrame([1, 2, 3]), None


class TestParamsTask(Task):


    __test__ = False


    def fit(self, df: pd.DataFrame, multiply: int) -> tuple[pd.DataFrame, TaskResult]:
        return df, None


    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
        return pd.DataFrame([1, 2, 3]), None


def test_pipeline():
    
    df = pd.DataFrame([1, 2, 3, 4])

    tasks = [FistTestTask()]

    pipeline = Pipeline(tasks)

    pipeline.fit(df)
    predict_df = pipeline.predict(df)

    assert len(predict_df) == 3


def test_params():

    df = pd.DataFrame([1, 2, 3, 4])
    pipeline = Pipeline([TestParamsTask()])

    with pytest.raises(Exception):
        pipeline.fit(df)

    pipeline.fit(df, multiply=7)
