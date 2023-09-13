from ..base.task import Task, TaskResult
from ..tasks.feature_generation import FeatureGenerationResult

import pandas as pd
from IPython.display import display
from typing import List, Dict

from sklearn.feature_selection import SelectKBest, SelectFromModel, SequentialFeatureSelector, VarianceThreshold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from tsfresh.feature_selection import select_features


class FeatureSelectionResult(TaskResult):
    """
    Represents the result of feature selection.

    Attributes:
        selected_features (list): List of selected feature names.
    """
    def __init__(self):
        self.selected_features = None

    def show(self) -> None:
        """
        Display the total selected features.
        """
        display(f"Total selected features: {len(self.selected_features)}")


class FeatureSelectionTask(Task):
    """
    Task for feature selection.

    Attributes:
        feature_generation_result (FeatureGenerationResult): The result of feature generation.
        target (str): The target feature name.
        feature_selection_method (str | None): Feature selection method. Allowed values are:
                - 'univariate': Perform univariate feature selection based on statistical tests.
                - 'tsfresh': Utilize the 'tsfresh' library for automated time series feature selection.
                - 'sequential': Sequential feature selection using an estimator (e.g., RandomForest) for classification or regression.
                - 'frommodel': Select features using an estimator (e.g., RandomForest) for classification or regression.
        feature_selection_method (str | None): Feature selection method.
        feature_selection_estimator (str | None): Feature selection estimator.
        remove_constant_features (bool): Whether to remove constant features.
    """

    feature_generation_result: FeatureGenerationResult

    def __init__(self, target: str, 
                 n_features_to_select: float | int | None = 0.2,
                 feature_selection_method: str | None = 'frommodel',
                 feature_selection_estimator: str | None = 'regressor',
                 remove_constant_features: bool = True,
                 ):
        """
        Initialize a FeatureSelectionTask.

        Args:
            target (str): The target feature name.
            n_features_to_select (float | int | None): Number of features to select.
            feature_selection_method (str | None): Feature selection method. Allowed values are:
                - 'univariate': Perform univariate feature selection based on statistical tests.
                - 'tsfresh': Utilize the 'tsfresh' library for automated time series feature selection.
                - 'sequential': Sequential feature selection using an estimator (e.g., RandomForest) for classification or regression.
                - 'frommodel': Select features using an estimator (e.g., RandomForest) for classification or regression.
            feature_selection_estimator (str | None): Feature selection estimator.
            remove_constant_features (bool): Whether to remove constant features.
        """
        # Check if feature_selection_method is valid
        if feature_selection_method not in ('univariate', 'tsfresh', 'sequential', 'frommodel'):
            raise ValueError("Invalid feature_selection_method. Allowed values are 'univariate', 'tsfresh', 'sequential', or 'frommodel'.")

        self.target = target
        self.remove_constant_features = remove_constant_features
        self.n_features_to_select = n_features_to_select
        self.feature_selection_method = feature_selection_method
        self.feature_selection_estimator = feature_selection_estimator
        super().__init__()


    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSelectionResult]:
        """
        Fit the feature selection model and select features.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            tuple[pd.DataFrame, FeatureSelectionResult]: A tuple containing the DataFrame with selected features
            and the FeatureSelectionResult.
        """
        # Copy the input DataFrame to avoid modifying the original data.
        df_copy = df.copy()

        # Convert all columns to numeric, handling errors by coercing to NaN.
        df_copy[df_copy.columns] = df_copy[df_copy.columns].apply(pd.to_numeric, errors='coerce')
        df_copy = df_copy.select_dtypes(include=['number', 'bool'])

        if self.remove_constant_features:
            # Remove constant features based on variance.
            constant_threshold = 0.0
            constant_filter = VarianceThreshold(threshold=constant_threshold)
            constant_filter.fit(df_copy)
            df_copy = df_copy[constant_filter.get_feature_names_out()]

        if self.target and self.target in df_copy.columns:
            if isinstance(self.n_features_to_select, float):
                # Calculate the number of features to select as a fraction of the total.
                num_features_to_select = int(df_copy.shape[1] * self.n_features_to_select)
            else:
                num_features_to_select = self.n_features_to_select

            # Fill NaN values with 0 for compatibility with feature selection methods.
            df_copy.fillna(0, inplace=True)

            if self.feature_selection_method == 'tsfresh':
                # Perform feature selection using tsfresh.
                _df_copy = select_features(df_copy.drop(columns=[self.target]), df_copy[self.target], fdr_level=0.05)
                selected_feature_names = _df_copy.columns

            if self.feature_selection_method == 'univariate':
                # Perform univariate feature selection.
                selector = SelectKBest(k=num_features_to_select)
                # Fit the feature selector to the data.
                selector.fit(df_copy.drop(columns=[self.target]), df_copy[self.target])
                selected_feature_names = selector.get_feature_names_out()

            elif self.feature_selection_method == 'sequential': # slow
                if self.feature_selection_estimator == 'regressor':
                    estimator = RandomForestRegressor(verbose=0, random_state=42)
                elif self.feature_selection_estimator == 'classifier':
                    estimator = RandomForestClassifier(verbose=0, random_state=42)
                else:
                    estimator = clone(self.feature_selection_estimator)
                selector = SequentialFeatureSelector(estimator, n_features_to_select=num_features_to_select)
                # Fit the feature selector to the data.
                selector.fit(df_copy.drop(columns=[self.target]), df_copy[self.target])
                selected_feature_names = selector.get_feature_names_out()

            elif self.feature_selection_method == 'frommodel':
                if self.feature_selection_estimator == 'regressor':
                    estimator = RandomForestRegressor(verbose=0, random_state=42)
                elif self.feature_selection_estimator == 'classifier':
                    estimator = RandomForestClassifier(verbose=0, random_state=42)
                else:
                    estimator = clone(self.feature_selection_estimator)
                selector = SelectFromModel(estimator, max_features=num_features_to_select)
                # Fit the feature selector to the data.
                selector.fit(df_copy.drop(columns=[self.target]), df_copy[self.target])
                selected_feature_names = selector.get_feature_names_out()
            
        # Create a FeatureSelectionResult instance to store the results.
        result = FeatureSelectionResult()
        
        # Create a DataFrame with the selected features and raw columns.
        if hasattr(self, 'feature_generation_result'):
            result.selected_features = list(set(selected_feature_names).intersection(self.feature_generation_result.generated_features))
            self.feature_generation_result.selected_features = result.selected_features
            df_result = df[self.feature_generation_result.raw_columns + result.selected_features]
        else:
            result.selected_features = selected_feature_names
            df_result = df[result.selected_features]

        

        return df_result, result

    def predict(self, df: pd.DataFrame, result: FeatureSelectionResult) -> tuple[pd.DataFrame, FeatureSelectionResult]:
        """
        Perform predictions.

        Args:
            df (pd.DataFrame): The input DataFrame.
            result (FeatureSelectionResult): The result of feature selection.

        Returns:
            tuple[pd.DataFrame, FeatureSelectionResult]: A tuple containing the DataFrame and the FeatureSelectionResult.
        """
        if hasattr(self, 'feature_generation_result'):
            df_result = df[self.feature_generation_result.raw_columns + result.selected_features]
        else:
            df_result = df[result.selected_features]
        return df_result, None
