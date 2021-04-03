from sklearn.model_selection import train_test_split
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ts_train_test_split(df, seq_len, 
                     points_ahead=1, gap=0, shag=1, intersection=True,
                     test_size=None,train_size=None, random_state=None, shuffle=True,stratify=None):
    """
	df - требование, но если тебе не хочется то просто сделай np.array на выходе и все
    Разбить временные ряды на трейн и тест
    seq_len- количество времменых точек в трейн
    points_ahead - количество времменых точек в прогнозе
    gap - расстояние между концом трейна и началом теста
    intersection - если нет, то в выборке нет перескающих множеств (временнызх моментов)
    shag - через сколько прыгаем
    train_size - float от 0 до 1
    
    
    
    
    """
    #TODO требования к входным данным прописать
    #TODO переписать энергоэффективно чтобы было
    #TODO пока временные характеристики int_ами пора бы в pd.TimdDelta
    # нет индексов 
    
    how='seq to seq'

# -------------------------------------------------------  
#             
# -------------------------------------------------------  


    x_start=0
    x_end= x_start + seq_len
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
        x_end= x_start + seq_len
        y_start = x_end + gap +1
        y_end = y_start + points_ahead
          
    
    if test_size==0:
        indices = np.array(range(len(X)))
        #             np.random.seed(random_state)
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]
        return X,y
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
		

		
def run_epoch(model, iterator, optimizer, criterion, points_ahead=1, phase='train'):
    is_train = (phase == 'train')
    if is_train:
        model.train()
    else:
        model.eval()
    
    epoch_loss = 0
    
    if points_ahead !=1:
        assert (points_ahead > 0) & (type(points_ahead)==type(int()))
        def forecast_multistep(model,y_pred,points_ahead):
            new_x = y_pred
            for j in range(points_ahead-1):
                new_x = model(new_x).unsqueeze(1)
                y_pred = torch.cat([y_pred,new_x],1)
            return y_pred
    else:
        def forecast_multistep(model,y_pred,points_ahead):
            return y_pred

    all_y_preds = []
    with torch.set_grad_enabled(is_train):
        for i, (x,y) in enumerate(iterator):
            x,y = np.array(x),np.array(y) #df.index rif of
            model.initHidden(x.shape[0])
            
            x = torch.tensor(x).float().to(device).requires_grad_()
            y_true = torch.tensor(y).float().to(device)
            y_pred = model(x).unsqueeze(1)
            y_pred = forecast_multistep(model,y_pred,points_ahead)
            if phase == 'forecast':
                all_y_preds.append(y_pred)
                continue # in case of pahse = 'forecast' criterion is None
                    
            loss = criterion(y_pred,y_true)
            if is_train:
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

            
            epoch_loss += loss.item()
    if phase != 'forecast':
        return epoch_loss / len(iterator)#, n_true_predicted / n_predicted
    else:
        return torch.cat(all_y_preds).detach().cpu().numpy()