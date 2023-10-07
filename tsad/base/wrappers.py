from ..base.task import Task
import pandas as pd


def SklearnWrapper(sklearnClass):

    """
    На данный момент с этим классом есть проблемы, так как sklearn_kwargs аргумент 
    может использоваться фактически двумя разными задачами этого врапера, потнециальное
    решение - фильтровать аргементы kwargs по принципу использования
    https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function 
    """
    
    class SklearnWrappedTask(Task):
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
            self.sklearnClass.fit(df,**sklearn_kwargs)
            new_df = self._sklearn_predict(df)
            return new_df
        
        def predict(self, df:pd.DataFrame):
            new_df = self._sklearn_predict(df)
            return new_df
            
            
    return SklearnWrappedTask