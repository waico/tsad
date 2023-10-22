from tsad.tasks.eda import HighLevelDatasetAnalysisTask, TimeDiscretizationTask
from tsad.tasks.eda import FindNaNTask, EquipmentDowntimeTask
from tsad.tasks.preprocess import ResampleProcessingTask 
from tsad.tasks.preprocess import SplitByNaNTask, PrepareSeqSamplesTask
from tsad.tasks.deep_learning_anomaly_detection import ResidualAnomalyDetectionTask
from tsad.tasks.deep_learning_forecasting import DeepLeaningTimeSeriesForecastingTask

from tsad.base.pipeline import Pipeline
from tsad.base.wrappers import SklearnWrapper

from sklearn.preprocessing import StandardScaler

StandardScalerTask = SklearnWrapper(StandardScaler)

multivatiateTimeSeriesDeepLearningForecastingTaskSet = [
    HighLevelDatasetAnalysisTask(),
    TimeDiscretizationTask(freq_tobe_approach='most_frequent'),
    FindNaNTask(),
    EquipmentDowntimeTask(),
    ResampleProcessingTask(),
    StandardScalerTask(),
    SplitByNaNTask(),
    PrepareSeqSamplesTask(len_seq=10),
    DeepLeaningTimeSeriesForecastingTask(),
]


ResidualAnomalyDetectionTaskSet = [
    HighLevelDatasetAnalysisTask(),
    TimeDiscretizationTask(freq_tobe_approach='most_frequent'),
    FindNaNTask(),
    EquipmentDowntimeTask(),
    ResampleProcessingTask(),
    StandardScalerTask(),
    SplitByNaNTask(),
    PrepareSeqSamplesTask(len_seq=10),
    ResidualAnomalyDetectionTask(),
]