from typing import List, Optional
import pandas as pd

from ..base.task import Task, TaskResult
from ..utils.visualization import plot_signals


class VisualizationTimeseriesResult(TaskResult):
    """
    Stores the result of time series visualization.

    Attributes:
        df (pd.DataFrame | None): The DataFrame containing time series data.
        features (List[str] | None): List of features to visualize (optional).
        fig (go.Figure | None): The Plotly figure for visualization.
    """

    def __init__(self,
                 features: List[str] | None = None,
                 use_resampler: bool = True,
                 scale: Optional[str] = None,
                 scale_columns: Optional[List[str]] = None,
                 show_fig: bool = True):
        """
        Initialize a VisualizationTimeseriesResult.

        Args:
            features (List[str], optional): List of feature names.
            use_resampler (bool, optional): Whether to use the plotly_resampler for interactive zooming.
            scale (str, optional): Scaling method for the signals. Options are 'minmax' and 'standard'.
            scale_columns (List[str], optional): List of specific columns to scale.
            show_fig (bool, optional): Whether to display the plot.

        """
        self.df: Optional[pd.DataFrame] = None
        self.features: Optional[List[str]] = features
        self.fig = None
        self.use_resampler = use_resampler
        self.scale = scale
        self.scale_columns = scale_columns
        self.show_fig = show_fig


    def show(self) -> None:
        """
        Display or process the generated features as needed.

        This method displays the total number of generated features.

        Example:
        result = TimeseriesVisualizationResult()
        result.df = df
        result.show()  # Display the time series plot.
        """
        if self.df is not None and self.features is None:
            self.fig = plot_signals(self.df, self.use_resampler, self.scale, self.scale_columns, self.show_fig)
        elif self.df is not None and self.features is not None:
            self.fig = plot_signals(self.df[self.features], self.use_resampler, self.scale, self.scale_columns, self.show_fig)

class VisualizationTimeseriesTask(Task):
    """
    Task for time series visualization.

    Attributes:
        features (List[str] | None): List of features to consider (optional).
        use_resampler (bool, optional): Whether to use the plotly_resampler for interactive zooming.
        scale (str, optional): Scaling method for the signals. Options are 'minmax' and 'standard'.
        scale_columns (List[str], optional): List of specific columns to scale.
        show_fig (bool, optional): Whether to display the plot.
    """
    def __init__(self, features: List[str] | None = None,
                 use_resampler: bool = True,
                 scale: Optional[str] = None,
                 scale_columns: Optional[List[str]] = None,
                 show_fig: bool = True):
        """
        Initialize a TimeseriesVisualizationTask.

        Args:
            features (List[str] | None): List of features to visualize (optional).
        """
        self.features = features
        self.use_resampler = use_resampler
        self.scale = scale
        self.scale_columns = scale_columns
        self.show_fig = show_fig

    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, VisualizationTimeseriesResult]:
        """
        Fit the time series visualization task to the input data.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            tuple[pd.DataFrame, VisualizationTimeseriesResult]: A tuple containing the DataFrame and the TimeseriesVisualizationResult.

        Example:
        task = TimeseriesVisualizationTask(features=['feature1', 'feature2'])
        df_fit, result = task.fit(df)
        """
        result = VisualizationTimeseriesResult(features=self.features,
                                                use_resampler=self.use_resampler,
                                                scale=self.scale,
                                                scale_columns=self.scale_columns,
                                                show_fig=self.show_fig)
        result.features = self.features
        result.df = df
        return df, result

    def predict(self, df: pd.DataFrame, result: VisualizationTimeseriesResult) -> pd.DataFrame:
        """
        Predict using the generated features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            result (VisualizationTimeseriesResult): The result of time series visualization.

        Returns:
            tuple[pd.DataFrame, VisualizationTimeseriesResult]: A tuple containing the DataFrame and the TimeseriesVisualizationResult.

        Example:
        df_predict, _ = task.predict(df_fit, result)
        """
        result = VisualizationTimeseriesResult(features=self.features,
                                                use_resampler=self.use_resampler,
                                                scale=self.scale,
                                                scale_columns=self.scale_columns,
                                                show_fig=self.show_fig)
        result.features = self.features
        result.df = df
        return df, result
