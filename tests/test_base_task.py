import pandas as pd

from tsad.base.task import Task, TaskStatus, TaskResult


class DummyTaskResult(TaskResult):
    def save(self) -> str:
        return "Dummy save"

    def show(self) -> None:
        pass


class DummyTask(Task):
    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
        return df, DummyTaskResult()

    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
        return df, DummyTaskResult()


def test_task_status_enum():
    assert TaskStatus.UNKNOWN.value == "UNKNOWN"
    assert TaskStatus.RUNNING.value == "RUNNING"
    assert TaskStatus.SUCCEEDED.value == "SUCCEEDED"
    assert TaskStatus.FAILED.value == "FAILED"


def test_task_result_save():
    result = DummyTaskResult()
    assert result.save() == "Dummy save"


def test_task_init():
    task = DummyTask()
    assert task.name is None


def test_task_fit():
    task = DummyTask()
    df = pd.DataFrame()
    result_df, result = task.fit(df)
    assert result_df.equals(df)
    assert isinstance(result, DummyTaskResult)


def test_task_predict():
    task = DummyTask()
    df = pd.DataFrame()
    result_df, result = task.predict(df)
    assert result_df.equals(df)
    assert isinstance(result, DummyTaskResult)
