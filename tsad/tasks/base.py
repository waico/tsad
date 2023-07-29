import abc
import pandas as pd

from enum import Enum


class TaskStatus(Enum):
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class TaskResult(abc.ABC):

    def save(self) -> str:
        pass

    @abc.abstractmethod
    def show(self) -> None:
        pass


class Task(abc.ABC):

    name: str
    status: TaskStatus

    def __init__(self, name: str | None = None):
        self.name = name

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
        pass
