import abc
import pandas as pd

from enum import Enum


class TaskStatus(Enum):

    """
    # TaskStatus

    Класс TaskStatus является перечислением, содержащим возможные статусы задачи.

    """
    UNKNOWN = "UNKNOWN"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class TaskResult(abc.ABC):

    """
    # Документация для класса TaskResult

    Класс TaskResult является абстрактным базовым классом, предназначенным для сохранения и отображения результатов задач.

    ### Методы:

    - save() -> str: абстрактный метод, возвращающий строку, содержащую сохраненные результаты задачи.
    - show() -> None: абстрактный метод, отображающий результаты задачи.
    """

    def save(self) -> str:
        pass

    @abc.abstractmethod
    def show(self) -> None:
        pass


class Task(abc.ABC):

    """
    # Документация для класса Task

    Класс Task является абстрактным базовым классом для задач, которые могут быть выполнены на наборе данных.

    ### Атрибуты:

    - name: str: имя задачи.
    - status: TaskStatus: текущий статус задачи.

    ### Методы:

    - __init__(name: str | None = None) -> None: конструктор класса, инициализирующий атрибуты name и status.
    - fit(df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]: абстрактный метод, выполняющий обучение задачи на наборе данных и возвращающий результаты обучения вместе с обновленным набором данных.
    - predict(df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]: абстрактный метод, выполняющий предсказание задачи на наборе данных и возвращающий результаты предсказания вместе с исходным набором данных.

    ### Пример использования:

    ```python
    class CustomTask(Task):
        def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
            # реализация обучения задачи
            result = TaskResult()
            # ...
            return df, result

        def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, TaskResult]:
            # реализация предсказания задачи
            result = TaskResult()
            # ...
            return df, result

    task = CustomTask("Моя задача")
    df = pd.DataFrame(...)
    output_df, result = task.fit(df)
    print(output_df)
    result.show()

    """

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
