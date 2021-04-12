from sklearn.model_selection import train_test_split
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ts_train_test_split(df, len_seq, 
                     points_ahead=1, gap=0, shag=1, intersection=True,
                     test_size=None,train_size=None, random_state=None, shuffle=True,stratify=None):
    """
	df - требование, но если тебе не хочется то просто сделай np.array на выходе и все
    Разбить временные ряды на трейн и тест
    len_seq- количество времменых точек в трейн
    points_ahead - количество времменых точек в прогнозе
    gap - расстояние между концом трейна и началом теста
    intersection - если нет, то в выборке нет перескающих множеств (временнызх моментов)
    shag - через сколько прыгаем
    train_size - float от 0 до 1
    
    return list of dfs
    
    
    
    
    """
    #TODO требования к входным данным прописать
    #TODO переписать энергоэффективно чтобы было
    #TODO пока временные характеристики int_ами пора бы в pd.TimdDelta
    # нет индексов 
    assert len_seq + points_ahead + gap + shag-1 <= len(df)
    how='seq to seq'

# -------------------------------------------------------  
#             
# -------------------------------------------------------  


    x_start=0
    x_end= x_start + len_seq
    y_start = x_end + gap +1
    y_end = y_start + points_ahead
    
    if intersection:
        # ради вычислительной нагрузки такой кастыль
        def compute_new_x_start(x_start,y_end,shag):
            return x_start + shag
    else:
        def compute_new_x_start(x_start,y_end,shag):
            y_end + shag
    
    X = []
    y = []
    while y_end <= len(df):
        X.append(df[x_start:x_end])
        y.append(df[y_start:y_end])
        
        x_start= compute_new_x_start(x_start,y_end,shag)
        x_end= x_start + len_seq
        y_start = x_end + gap +1
        y_end = y_start + points_ahead
          
    
    if test_size==0:
        indices = np.array(range(len(X)))
        #             np.random.seed(random_state)
        if shuffle:
            np.random.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]
        return X,[],y,[]
    else:
        return train_test_split(X,y, 
                                test_size=test_size, 
                                train_size=train_size, 
                                random_state=random_state, 
                                shuffle=shuffle, 
                                stratify=stratify
                               )
						   
						   
						   
class Loader:
    def __init__(self, X,y, batch_size,shuffle=True):
        if shuffle==True:
            indices = np.array(range(len(X)))
            np.random.shuffle(indices)
            self.X = X[indices]
            self.y = y[indices]
        else:
            self.X = X
            self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        self.i = - self.batch_size
        return self

    def __next__(self):
        if self.i+self.batch_size < len(self.X):
            self.i+=self.batch_size
            return self.X[self.i:self.i+self.batch_size], self.y[self.i:self.i+self.batch_size]
        elif self.i+2*self.batch_size < len(self.X):        
            return self.X[self.i:], self.y[self.i:]
        else:
            raise StopIteration
            
    def __len__(self):
        return len(np.arange(0,len(self.X),self.batch_size))
		        
        
def split_by_repeated(series,df=None):
    """
    retrun dict with lists of ts whwre keys is unique values
    """
    series = series.copy().dropna()
    if len(series.unique())==1:
        result = {series.unique()[0]:series}
    elif len(series.unique())>1:
        result = {uni:[] for uni in series.unique()}
        recent_i=0
        recent_val=series.values[0]
        for i in range(len(series)):
            val = series.values[i]
            if (recent_val == val):
                continue
            else:
                result[recent_val].append(series[recent_i:i])
                recent_i=i
                recent_val = val

        if i == len(series)-1:
            if (recent_val == val):
                result[recent_val].append(series[recent_i:i+1])
            else:
                result[recent_val].append(series[recent_i:i+1])
    else:
        raise NameError('0 series')


    if df is not None:
        new_result = {uni:[] for uni in series.unique()}
        for key in result:
            for i in range(len(result[key])):
                if len(result[key][i]) <=1:
                    continue
                else:
                    new_result[key].append(df.loc[result[key][i].index])
        return new_result
    else:
        return result