from ..base.task import Task, TaskResult
from ..tasks.feature_generation import FeatureGenerationResult

import pandas as pd
from IPython.display import display
from typing import List, Dict

from sklearn.feature_selection import SelectKBest, SelectFromModel, SequentialFeatureSelector, VarianceThreshold
from sklearn.base import clone
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import re


class FeatureSelectionResult(TaskResult):
    
    def __init__(self):
        
        self.selected_features = None

    def show(self) -> None:

        display(f"Total selected features: {len(self.selected_features)}")


class FeatureSelectionTask(Task):

    feature_generation_result: FeatureGenerationResult

    def __init__(self, target: str, 
                 n_features_to_select: float | int | None = 0.2,
                 feature_selection_method: str | None = 'univariate',
                 feature_selection_estimator: str | None = 'regressor',
                 remove_constant_features: bool = True,
                 ):
        self.target = target
        self.remove_constant_features = remove_constant_features
        self.n_features_to_select = n_features_to_select
        self.feature_selection_method = feature_selection_method
        self.feature_selection_estimator = feature_selection_estimator
        super().__init__()

    
    def fit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSelectionResult]:
        
        df_prep = df.copy()
        df_prep[df_prep.columns] = df_prep[df_prep.columns].apply(pd.to_numeric, errors='coerce')
        
        if self.remove_constant_features:
            constant_filter = VarianceThreshold(threshold=0)
            constant_filter.fit(df_prep)
            df_prep = df_prep[constant_filter.get_feature_names_out()]
            
        if self.target and self.target in df_prep.columns:
            if isinstance(self.n_features_to_select, float):
                self.n_features_to_select = int(df_prep.shape[1] * self.n_features_to_select)
            
            df_prep.fillna(0, inplace=True)

            if self.feature_selection_method == 'univariate':
                selector = SelectKBest(k=self.n_features_to_select)

            elif self.feature_selection_method == 'sequential': # slow
                if self.feature_selection_estimator == 'regressor':
                    estimator = RandomForestRegressor(verbose=0)
                elif self.feature_selection_estimator == 'classifier':
                    estimator = RandomForestClassifier(verbose=0)
                else:
                    estimator = clone(self.feature_selection_estimator)
                selector = SequentialFeatureSelector(estimator, n_features_to_select=self.n_features_to_select)

            else:  # 'frommodel'
                if self.feature_selection_estimator == 'regressor':
                    estimator = RandomForestRegressor(verbose=0)
                elif self.feature_selection_estimator == 'classifier':
                    estimator = RandomForestClassifier(verbose=0)
                else:
                    estimator = clone(self.feature_selection_estimator)
                selector = SelectFromModel(estimator, max_features=self.n_features_to_select)
            
            selector.fit(df_prep.drop(columns=[self.target]), df_prep[self.target])
        
        df_prep = df[list(selector.get_feature_names_out()) + self.feature_generation_result.raw_columns]

        result = FeatureSelectionResult()
        result.selected_features = list(set(selector.get_feature_names_out()).intersection(self.feature_generation_result.generated_features))
        self.feature_generation_result.selected_features = result.selected_features
        
        return df[self.feature_generation_result.raw_columns + result.selected_features], result

    def predict(self, df: pd.DataFrame, result: FeatureSelectionResult) -> tuple[pd.DataFrame, FeatureSelectionResult]:

        return df, None


    
    
    
    
    

