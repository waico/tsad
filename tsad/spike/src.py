class R_to_I():
    """
    A class that allows you to make a binary multidimensional time series from a 
    multidimensional time series of Real values. It is needed to adapt the 
    input for spike neural networks. API uses skirit-learn templates
    
    [Ru]: Класс который позволяет из многомерного временного ряда реальных значений
    сделать бинарный многомерный врменной ряд. Нужен для адаптации входных для спайковых 
    нейронных сетей. Для API используются шаблоны skirit-learn
    """
    
    def koefI_to_koefSeqI(self,ts):
        """
        A function that converts the number 5 to 5 consecutive impulses in frames
        
        Parameters
        ----------
        ts : pd.Series 
            The initial series with Real numbers
        
        Returns
        -------
        new_ts : pd.Series 
            The output series with binary labels
        
        Функция которая преобрарузет число 5 в 5 последовательных импульсов 
        """
        buffer = 0
        new_array = []
        for ind,val in ts.iteritems():
            if val !=0:
                buffer += val 
            new_val = 1 if buffer > 0 else 0
            new_array.append(new_val)
            buffer = buffer - 1 if buffer > 0 else 0
        new_ts = pd.Series(new_array,index=ts.index)
        return new_ts
    
    
    def __init__(self,verbose=False):
        """
        A class that allows you to make a binary multidimensional time series from a 
        multidimensional time series of Real values. It is needed to adapt the 
        input for spike neural networks. API uses skirit-learn templates. Useful if the signal 
        is being written by aperture.

        [Ru]: Класс который позволяет из многомерного временного ряда реальных значений
        сделать бинарный многомерный врменной ряд. Нужен для адаптации входных для спайковых 
        нейронных сетей. Для API используются шаблоны skirit-learn. Полезен если сигнал пишется
        по апертуре. 
        
        Parameters
        ----------
        verbose : boolean, default = False
            If True, additional infomation will be provided
        """
        self.verbose = verbose
    
    def _check_input_(self,x):
        assert any(np.isnan(x))
        assert len(x)!=0
        x.columns = x.columns.astype(str)
        
    def fit(self,x):
        """
        Fit the model with x

        [Ru]: Выделение апертуры в соотвествии с х
        
        Parameters
        ----------
        x : pd.DataFrame
            Training data. 
        """
        self._check_input_(x)      
        df = x.dropna().diff().dropna().abs()
        vector_koef = []
        self.block_columns = [] #колонки, которые не меняются вообще, то есть константы, их не будет после 
                                #трансформации, а после inverse_transform они будут константой той же
        for col in df:
            ts = df[col]
            ts = ts[ts!=0.]
            if len(ts)!=0:
                vector_koef.append(ts.value_counts().index[0])
            else: 
                self.block_columns.append(col)
        self.all_columns = x.columns
        self.norm_columns =  self.all_columns[~self.all_columns.isin(self.block_columns)]
        self.vector_koef = pd.DataFrame(np.array(vector_koef).reshape(1,-1), 
                                        columns = self.norm_columns)
        return self
    
    def transform(self,x):
        """
        Converting real values x to binary.

        [Ru]: Преобразование реальных значений в бианрные. 
        
        Parameters
        ----------
        x : pd.DataFrame
            Data which we want to transform from real values to binary values. 
        """
        self._check_input_(x)  
        result= x.copy()
        self.vector_0 = result.iloc[0:1] 
        self.index = result.index   
        result = result[self.norm_columns]
        result = (result.diff() / self.vector_koef.values).dropna().round().astype(int)        
        print('Fistt stage done')
        # разделяем на положительные и отрицательные признаки, посути увеличив количесто признаков в 2 раза 
        result = pd.concat([result[result>=0].add_suffix('_pos'),
                            result[result<=0].add_suffix('_neg')] ,
                            axis=1).fillna(0).astype(int).abs()
        print('Second stage done')
        if self.verbose:
            print('Максимальные величины импульсов (не единичные еще) для признаков')
            display_df(result.max())
            print('Распределение величин импульсов по всей таблице')
            display_df(pd.Series(result.values.ravel()).value_counts())
        print('Thied stage done')
        #  Функция которая преобрарузет число 5 в 5 последовательных импульсов 
        result = result.apply(self.koefI_to_koefSeqI)
        print('thourh stage done')
        return result # первый нан
    
    def fit_trasform(self,x):
        """
        Fitting and converting real values x to binary.

        [Ru]: Преобразование реальных значений в бианрные. 
        
        Parameters
        ----------
        x : pd.DataFrame
            Data which we want to transform from real values to binary values. 
        """
        
        return self.fit(x).transform(x)
    
    def inverse_tranform(self,x):
        """
        Fitting and converting real values x to binary.

        [Ru]: Преобразование реальных значений в бианрные. 
        
        Parameters
        ----------
        x : pd.DataFrame
            Data which we want to transform from binary values to real values.
            
        Notes
        ----------
        Очень важно иметь общие индекс с transform(self,x)
        """        
        self._check_input_(x)
        result = x.copy()
        neg = result.filter(regex='_neg',axis=1)
        neg.columns = neg.columns.str.strip('_neg')
        result = result.filter(regex='_pos',axis=1).values - neg
        del neg
        print('five stage done')
        result = result *  self.vector_koef.values.astype(int)
        result = pd.concat([self.vector_0,result])
        assert (result.index == self.index).all()
        result = result.cumsum()
        
        for col in self.block_columns:
            result[col] = self.vector_0[col].values[0]
        result = result[self.all_columns]
        return result