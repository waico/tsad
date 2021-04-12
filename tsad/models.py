from torch import nn, optim
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np


class SimpleLSTM(nn.Module):
    def __init__(self, in_features, n_hidden, n_layers=2, bidirectional=False, dropout=0.5):
        super(SimpleLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=in_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            dropout=dropout,
                            batch_first =True,
                            bidirectional = bidirectional
                           )
        self.k_bidir = 2 if bidirectional else 1 

        self.linear = nn.Linear(in_features=n_hidden, out_features=in_features)
    
    
    def initHidden(self,batch_size):
        self.hidden = (
            torch.zeros(self.n_layers*self.k_bidir, batch_size, self.n_hidden).to(device),
            torch.zeros(self.n_layers*self.k_bidir, batch_size, self.n_hidden).to(device)
        )
    def forward(self, sequences):
        batch_size  = len(sequences)
        lstm_out, self.hidden = self.lstm(sequences, self.hidden )
        last_time_step = lstm_out.reshape(-1, batch_size, self.n_hidden)[-1] # -1 is len_seq

        y_pred = self.linear(last_time_step)
        return y_pred

    def run_epoch(self, iterator, optimizer, criterion, points_ahead=1, phase='train',encod_decode_model=False):
        is_train = (phase == 'train')
        if is_train:
            self.train()
        else:
            self.eval()
        
        epoch_loss = 0
        
        if points_ahead !=1:
            assert (points_ahead > 0) & (type(points_ahead)==type(int()))
            def forecast_multistep(y_pred,points_ahead):
                new_x = y_pred
                for j in range(points_ahead-1):
                    new_x = self.forward(new_x).unsqueeze(1)
                    y_pred = torch.cat([y_pred,new_x],1)
                return y_pred
        else:
            def forecast_multistep(y_pred,points_ahead):
                return y_pred

        all_y_preds = []
        with torch.set_grad_enabled(is_train):
            for i, (x,y) in enumerate(iterator):
                x,y = np.array(x),np.array(y) #df.index rif of
                self.initHidden(x.shape[0])
                
                x = torch.tensor(x).float().to(device).requires_grad_()
                y_true = torch.tensor(y).float().to(device)
                y_pred = self.forward(x).unsqueeze(1)
                y_pred = forecast_multistep(y_pred,points_ahead)
                
                if encod_decode_model:
                    y_pred = torch.cat([y_pred[:,i,:].unsqueeze(1) for i in range(y_pred.shape[1]-1,-1,-1)],1)
                
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






####################
#=======================================================
#              From https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED/blob/master/utils/matrix_generator.py
#=======================================================







#=======================================================
#              From other sourse
#=======================================================
# class ConvLSTMCell(nn.Module):

#     def __init__(self, input_dim, hidden_dim, kernel_size, bias):
#         """
#         MIT license
#         Authors and any other relevant information you can see:  
#         https://github.com/ndrplz/ConvLSTM_pytorch 
        
        
#         Initialize ConvLSTM cell.
#         Parameters
#         ----------
#         input_dim: int
#             Number of channels of input tensor.
#         hidden_dim: int
#             Number of channels of hidden state.
#         kernel_size: (int, int)
#             Size of the convolutional kernel.
#         bias: bool
#             Whether or not to add the bias.
#         """

#         super(ConvLSTMCell, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.kernel_size = kernel_size
#         self.padding = kernel_size[0] // 2, kernel_size[1] // 2
#         self.bias = bias

#         self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
#                               out_channels=4 * self.hidden_dim,
#                               kernel_size=self.kernel_size,
#                               padding=self.padding,
#                               bias=self.bias)

#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state

#         combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

#         combined_conv = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)

#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)

#         return h_next, c_next

#     def init_hidden(self, batch_size, image_size):
#         height, width = image_size
#         return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
#                 torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# class ConvLSTM(nn.Module):

#     """
#     MIT license
#     Authors and any other relevant information you can see:  
#     https://github.com/ndrplz/ConvLSTM_pytorch 
    
    
    
#     Parameters:
#         input_dim: Number of channels in input
#         hidden_dim: Number of hidden channels
#         kernel_size: Size of kernel in convolutions
#         num_layers: Number of LSTM layers stacked on each other
#         batch_first: Whether or not dimension 0 is the batch or not
#         bias: Bias or no bias in Convolution
#         return_all_layers: Return the list of computations for all layers
#         Note: Will do same padding.
#     Input:
#         A tensor of size B, T, C, H, W or T, B, C, H, W
#     Output:
#         A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
#             0 - layer_output_list is the list of lists of length T of each output
#             1 - last_state_list is the list of last states
#                     each element of the list is a tuple (h, c) for hidden state and memory
#     Example:
#         >> x = torch.rand((32, 10, 64, 128, 128))
#         >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
#         >> _, last_states = convlstm(x)
#         >> h = last_states[0][0]  # 0 for layer index, 0 for h index
#     """

#     def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
#                  batch_first=False, bias=True, return_all_layers=False):
#         super(ConvLSTM, self).__init__()

#         self._check_kernel_size_consistency(kernel_size)

#         # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
#         kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
#         hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
#         if not len(kernel_size) == len(hidden_dim) == num_layers:
#             raise ValueError('Inconsistent list length.')

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers
#         self.batch_first = batch_first
#         self.bias = bias
#         self.return_all_layers = return_all_layers

#         cell_list = []
#         for i in range(0, self.num_layers):
#             cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

#             cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
#                                           hidden_dim=self.hidden_dim[i],
#                                           kernel_size=self.kernel_size[i],
#                                           bias=self.bias))

#         self.cell_list = nn.ModuleList(cell_list)

#     def forward(self, input_tensor, hidden_state=None):
#         """
#         Parameters
#         ----------
#         input_tensor: todo
#             5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
#         hidden_state: todo
#             None. todo implement stateful
#         Returns
#         -------
#         last_state_list, layer_output
#         """
#         if not self.batch_first:
#             # (t, b, c, h, w) -> (b, t, c, h, w)
#             input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

#         b, _, _, h, w = input_tensor.size()

#         # Implement stateful ConvLSTM
#         if hidden_state is not None:
#             raise NotImplementedError()
#         else:
#             # Since the init is done in forward. Can send image size here
#             hidden_state = self._init_hidden(batch_size=b,
#                                              image_size=(h, w))

#         layer_output_list = []
#         last_state_list = []

#         seq_len = input_tensor.size(1)
#         cur_layer_input = input_tensor

#         for layer_idx in range(self.num_layers):

#             h, c = hidden_state[layer_idx]
#             output_inner = []
#             for t in range(seq_len):
#                 h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
#                                                  cur_state=[h, c])
#                 output_inner.append(h)

#             layer_output = torch.stack(output_inner, dim=1)
#             cur_layer_input = layer_output

#             layer_output_list.append(layer_output)
#             last_state_list.append([h, c])

#         if not self.return_all_layers:
#             layer_output_list = layer_output_list[-1:]
#             last_state_list = last_state_list[-1:]

#         return layer_output_list, last_state_list

#     def _init_hidden(self, batch_size, image_size):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
#         return init_states

#     @staticmethod
#     def _check_kernel_size_consistency(kernel_size):
#         if not (isinstance(kernel_size, tuple) or
#                 (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
#             raise ValueError('`kernel_size` must be tuple or list of tuples')

#     @staticmethod
#     def _extend_for_multilayer(param, num_layers):
#         if not isinstance(param, list):
#             param = [param] * num_layers
#         return param