#  Требования
#  работа как с одномерным pd.DataFrame так и с многомерными
# Наличие show_figure
# Наличие методов fit, predict, fit_predict
# Сохрание в атрибуты статистик и пределов: ucl, lcl, statistic



#
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

"""
nan - недопустимы
"""
class Hotelling():
    def __init__(self,koef_ucl=3):
        self.koef_ucl = koef_ucl
    
    def fit(self,df):
        if df.shape[1]==1:
            self.inv_cov = np.array(1/ np.cov(df.T)).reshape(1,1)
        else:
            try: 
                self.inv_cov =  np.linalg.inv(np.cov(df.T))
            except:
                self.inv_cov =  np.linalg.pinv(np.cov(df.T))
        self.mean = df.mean()
        statistic  = (((df - self.mean).values @ self.inv_cov) @ (df - self.mean).values.T).diagonal()
        self.ucl = statistic.mean()+self.koef_ucl*statistic.std()
        self.lcl = None
    
    def predict(self,df,show_figure=False):
        self.statistic  = pd.Series((((df - self.mean).values @ self.inv_cov) @ (df - self.mean).values.T).diagonal(),
                                 index=df.index
                                )
        anomalies = self.statistic[self.statistic>=self.ucl].index
        if show_figure:
            plt.figure()
            plt.plot(self.statistic,label='Hotelling statistic')
            plt.axhline(self.ucl,label='UCL',c='pink')
            for anom in anomalies:
                plt.axvline(anom,c='pink')
            plt.axvline(anom,c='pink',label=f'Anomalies, total {len(anomalies)} events')
            plt.xlabel('Datetime')
            plt.ylabel('Hotelling statistic')
            plt.title('Аномалии в нормальном режиме')
            plt.xticks(rotation=30)
            plt.legend()
            plt.show()            
        return anomalies
    
    
    def feature_importances(self,df):
        if not('ucl' in dir(self)):
            raise NameError("Fitting must be perfomed")
        feat_impor = []
        for col in df:
            _df = df.copy()
            _df[:] = 0
            _df[col] = (df - self.mean)[col]
            feat_impor.append(pd.Series(((_df.values @ self.inv_cov) @ _df.values.T).diagonal(),
                                        index=df.index) )
        feat_impor = pd.concat(feat_impor,1)#.rename(columns=df.columns)
        # нормировочка 
        _sum = feat_impor.sum(1).values
        for col in feat_impor:
            feat_impor[col] = (feat_impor[col].values / _sum) *100
        feat_impor.columns = df.columns
        return feat_impor
    
    def fit_predict(self,df,show_figure=False):
        self.fit(df)
        return self.predict(df,show_figure=show_figure)


