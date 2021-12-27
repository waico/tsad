from torch import nn, optim
import torch
from .fit import set_determenistic
import numpy as np

class mlp(nn.Module):
    def __init__(self, in_features, n_hidden, seed=None):
        
        set_determenistic(seed)
        super().__init__()
        
        
        self.in_features = in_features
        n_middle= int((in_features - n_hidden)/2) + n_hidden
        self.linear1 = nn.Linear(in_features=in_features, out_features=n_middle)
        self.linear2 = nn.Linear(in_features=n_middle, out_features=n_hidden) 
        self.linear3 = nn.Linear(in_features=n_hidden, out_features=n_middle) 
        self.linear4 = nn.Linear(in_features=n_middle, out_features=in_features) 
        self.Sigmoid = nn.Sigmoid()
    
    

    def forward(self, x):
        x = self.Sigmoid(self.linear1(x))
        x = self.Sigmoid(self.linear2(x))
        x = self.Sigmoid(self.linear3(x))
        y_pred = self.linear4(x)
        return y_pred

    def run_epoch(self, iterator, optimizer, criterion, points_ahead=1, phase='train', device=torch.device('cuda:0')):
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
            
            
            
            
            
            
class lstm(nn.Module):
    def __init__(self, in_features, n_hidden, seed=None):        
        super().__init__()
        set_determenistic(seed)
    
        n_middle= int((in_features - n_hidden)/2) + n_hidden
        
        self.in_features = in_features
        self.n_hidden = n_hidden
        self.n_middle = n_middle
        
        
        self.lstm1 = nn.LSTM(input_size=in_features,
                            hidden_size=n_middle,
                            batch_first =True)
        self.lstm2 = nn.LSTM(input_size=n_middle,
                            hidden_size=n_hidden,
                            batch_first =True)
        self.lstm3 = nn.LSTM(input_size=n_hidden,
                            hidden_size=n_middle,
                            batch_first =True)
        self.lstm4 = nn.LSTM(input_size=n_middle,
                            hidden_size=in_features,
                            batch_first =True)


        self.linear = nn.Linear(in_features=in_features, out_features=in_features)        
    
    
    def initHidden(self,batch_size,device):
        self.hidden_lstm1 = (
            torch.zeros(1, batch_size, self.n_middle).to(device),
            torch.zeros(1, batch_size, self.n_middle).to(device)
                            )
        self.hidden_lstm2 = (
            torch.zeros(1, batch_size, self.n_hidden).to(device),
            torch.zeros(1, batch_size, self.n_hidden).to(device)
                            )
        self.hidden_lstm3 = (
            torch.zeros(1, batch_size, self.n_middle).to(device),
            torch.zeros(1, batch_size, self.n_middle).to(device)
                            )
        self.hidden_lstm4 = (
            torch.zeros(1, batch_size, self.in_features).to(device),
            torch.zeros(1, batch_size, self.in_features).to(device)
                            )
    def forward(self, sequences):
        batch_size  = len(sequences)
        lstm_out1, self.hidden_lstm1 = self.lstm1(sequences, self.hidden_lstm1)
        lstm_out2, self.hidden_lstm2 = self.lstm2(lstm_out1, self.hidden_lstm2)
        lstm_out3, self.hidden_lstm3 = self.lstm3(lstm_out2, self.hidden_lstm3)
        lstm_out4, self.hidden_lstm4 = self.lstm4(lstm_out3, self.hidden_lstm4)
        
        # last_time_step = lstm_out4.reshape(-1, batch_size, self.in_features)[-1] # -1 is len_seq
        last_time_step = lstm_out4[:,-1,:]
        y_pred = self.linear(last_time_step)
        return y_pred

    def run_epoch(self, iterator, optimizer, criterion, phase='train', device=torch.device('cuda:0'), encod_decode_model=False, points_ahead=None):
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
                x,y = np.array(x),np.array(x)[:,-1,:] #!!! тут суть, см y
                self.initHidden(x.shape[0],device=device)
                
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