from ..base.task import Task
import pandas as pd


"""
На данный момент с этим классом есть проблемы, так как sklearn_kwargs аргумент 
может использоваться фактически двумя разными задачами этого врапера, потнециальное
решение - фильтровать аргементы kwargs по принципу использования
https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function 
"""

def SklearnWrapper(sklearnClass):
    """
    A decorator that wraps a scikit-learn class and returns a new TSAD Task .

    Parameters:
    ---------
    sklearnClass : class
        A scikit-learn class to be wrapped.

    Returns:
        class: A TSAD Task class that inherits from Task and wraps the scikit-learn class.
    """
    
    class SklearnWrappedTask(Task):
        """
        A TSAD Task class that inherits from Task and wraps the scikit-learn class.
        """

        def __init__(self,sklearnClass=sklearnClass,**kwargs):
            self.sklearnClass = sklearnClass(**kwargs)
            
        def _sklearn_predict(self,df):
            if 'predict' in dir(self.sklearnClass):
                new_df = self.sklearnClass.predict(df)
            elif 'transform' in dir(self.sklearnClass):
                new_df = self.sklearnClass.transform(df)
            else: 
                raise Exception('It is not appropriate sklearn class')
            new_df = pd.DataFrame(new_df,index=df.index,columns=df.columns)
            return new_df
            
            
        def fit_predict(self,df:pd.DataFrame,sklearn_kwargs={}):
            """
            Fits the wrapped scikit-learn class to the input data and returns the predicted values.

            Parameters:
            ----------
            df : pd.DataFrame
                The input data to fit the scikit-learn class.

            sklearn_kwargs : dict, optional
                Keyword arguments to pass to the scikit-learn fit method.

            Returns:
            -------
            pred : pd.DataFrame
                The predicted values from the scikit-learn class.
            """
            self.sklearnClass.fit(df,**sklearn_kwargs)
            new_df = self._sklearn_predict(df)
            return new_df
        
        def predict(self, df:pd.DataFrame):
            """
            Returns the predicted values from the wrapped scikit-learn class.

            Parameters:
            ----------
            df : pd.DataFrame
                The input data to fit the scikit-learn class.

            Returns:
            -------
            pred : pd.DataFrame
                The predicted values from the scikit-learn class.
            """
            new_df = self._sklearn_predict(df)
            return new_df
            
            
    return SklearnWrappedTask