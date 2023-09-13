from ..base.task import Task
import pandas as pd


def SklearnWrapper(sklearnClass):
    
    class SklearnWrappedTask(Task):
        def __init__(self,sklearnClass=sklearnClass,**kwargs):
            if '__sklearn_clone__' not in dir(sklearnClass):
                raise Exception('It is not sklearn class') 
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
            
            
        def fit(self,df:pd.DataFrame,**kwargs):
            
            self.sklearnClass.fit(df,**kwargs)
            new_df = self._sklearn_predict(df)
            return new_df
        
        def predict(self, df:pd.DataFrame):
            new_df = self._sklearn_predict(df)
            return new_df
            
            
    return SklearnWrappedTask