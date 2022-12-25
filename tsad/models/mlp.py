from torch import nn, optim
import torch
from .fit import set_determenistic
import numpy as np



class MLP(nn.Module):
    def __init__(self, tuple_layers, dropout=0.5, seed=None):
        set_determenistic(seed)
        super(MLP, self).__init__()
        
        if len(tuple_layers)<3:
            raise
        self.seq = nn.Sequential()
        layers = (nn.Linear(in_features=tuple_layers[i], 
                           out_features=tuple_layers[i+1])  for i in range(len(tuple_layers)-2))
        drops = (nn.Dropout(p=dropout) for i in range(len(tuple_layers)-2))
        sigmoids = ( nn.Sigmoid()      for i in range(len(tuple_layers)-2))
        seq = (j for sub in tuple(zip(layers,sigmoids,drops)) for j in sub)
        self.seq = nn.Sequential(*seq,
                                  nn.Linear(in_features=tuple_layers[-2], 
                                            out_features=tuple_layers[-1]))
    def forward(self, x):
        y  = self.seq(x)
        return y

    def run_epoch(self, iterator, optimizer, criterion, points_ahead=1, phase='train', 
                  device=torch.device('cuda:0'), encod_decode_model=False):
        self.to(device)
        
        is_train = (phase == 'train')
        if is_train:
            self.train()
        else:
            self.eval()
        
        epoch_loss = 0

        all_y_preds = []
        with torch.set_grad_enabled(is_train):
            for i, (x,y) in enumerate(iterator):
                x,y = np.array(x),np.array(y) #df.index rif of
    
                x = torch.tensor(x).float().to(device).requires_grad_()
                y_true = torch.tensor(y).float().to(device)
                y_pred = self.forward(x)
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