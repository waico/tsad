from torch import nn, optim
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleLSTM(nn.Module):
  def __init__(self, in_features, n_hidden, n_layers=2):
    super(SimpleLSTM, self).__init__()
    self.n_hidden = n_hidden
    self.n_layers = n_layers
    self.lstm = nn.LSTM(input_size=in_features,
                        hidden_size=n_hidden,
                        num_layers=n_layers,
                        dropout=0.5,
                        batch_first =True
                       )
    
    self.linear = nn.Linear(in_features=n_hidden, out_features=in_features)
    
    
  def initHidden(self,batch_size):
    self.hidden = (
        torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device),
        torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
    )
  def forward(self, sequences):
    batch_size  = len(sequences)
    lstm_out, self.hidden = self.lstm(sequences, self.hidden )
    last_time_step = lstm_out.reshape(-1, batch_size, self.n_hidden)[-1] # -1 is len_seq

    y_pred = self.linear(last_time_step)
    return y_pred