#  Требования
#  работа как с одномерным pd.DataFrame так и с многомерными
# Наличие show_figure
# Наличие методов fit, predict, fit_predict
# Сохрание в атрибуты статистик и пределов.



#
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class hotteling():
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
    
    def predict(self,df,show_figure=False):
        self.statistic  = pd.Series((((df - self.mean).values @ self.inv_cov) @ (df - self.mean).values.T).diagonal(),
                                 index=df.index
                                )
        anomalies = self.statistic[self.statistic>=self.ucl].index
        if show_figure:
            plt.figure()
            plt.plot(self.statistic,label='Hotteling statistic')
            plt.axhline(self.ucl,label='UCL',c='pink')
            for anom in anomalies:
                plt.axvline(anom,c='pink')
            plt.axvline(anom,c='pink',label=f'Anomalies, total {len(anomalies)} events')
            plt.xlabel('Datetime')
            plt.ylabel('Hotteling statistic')
            plt.title('Аномалии в нормальном режиме')
            plt.legend()
            plt.show()  
        return anomalies
    
    def fit_predict(self,df,show_figure=False):
        self.fit(df)
        return self.predict(df,show_figure=show_figure)


