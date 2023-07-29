from IPython import display
import torch
from ..iterators import Loader
from matplotlib import pyplot as plt
import numpy as np
import random

#     criterion = weighted_mse_loss
#     optimizer = optim.Adam(net.parameters(), lr=lr)
#     model = model()

def set_determenistic(seed=None,precision=10):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    #torch.set_printoptions(precision=precision)


def fit(
    model,
    optimiser,
    criterion,
    res_train_test_split,
    n_epochs = 10,
    batch_size = 2056,
    best_model_file =  './best_ae.pth', #None
    points_ahead = 1,
    random_state = None,
    show_progress = True,
    title=None,
    
       ):
    
    X_train, X_test, y_train, y_test = res_train_test_split[0],\
    res_train_test_split[1], res_train_test_split[2],res_train_test_split[3] 
    #     all_loader = Loader(scaled_research_df, scaled_research_df, batch_size, shuffle=False)
    train_iterator = Loader(X_train, y_train, batch_size, shuffle=True, random_state=random_state) # !!! кастыль
    val_iterator = Loader(X_test, y_test, batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    history_train = []
    history_val = []
    best_val_loss = float('+inf')
    show_progress_text =""
    for epoch in range(n_epochs):
        train_loss = model.run_epoch(train_iterator, optimiser, criterion, phase='train',
                                        points_ahead = points_ahead, device=device)  # , writer=writer)
        val_loss = model.run_epoch(val_iterator, None, criterion, phase='val',
                                        points_ahead = points_ahead, device=device)  # , writer=writer)

        history_train.append(train_loss)
        history_val.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_file is not None:
                torch.save(model.state_dict(), best_model_file)

        display.clear_output(wait=True)
        plt.figure()
        plt.plot(history_train, label=f'Train  {round(train_loss,3)}')
        plt.plot(history_val, label=f'Val  {round(val_loss,3)}')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.grid()
        plt.show()
        if show_progress:
            show_progress_text = f'Epoch: {epoch + 1:02} \n' + \
                                 f'\tTrain Loss: {train_loss:.3f} \n' + \
                                 f'\t Val. Loss: {val_loss:.3f} \n\n' +  \
                                 show_progress_text
            print(show_progress_text)

    if best_model_file is not None:
        model.load_state_dict(torch.load(best_model_file))
    return model.run_epoch(val_iterator, None, criterion, phase='val', device=device) 