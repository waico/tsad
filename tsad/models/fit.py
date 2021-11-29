from IPython import display
import torch
from ..src.useful import Loader
from matplotlib import pyplot as plt
import numpy as np
import random



def set_determenistic(seed=None,precision=10):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.set_printoptions(precision=precision)


def fit(
    model,
    optimiser,
    criterion,
    res_train_test_split,
    n_epochs = 10,
    batch_size = 2056,
    best_model_file =  './best_ae.pth',
       ):
    
    X_train, X_test, y_train, y_test = res_train_test_split[0],\
    res_train_test_split[1], res_train_test_split[2],res_train_test_split[3] 
    #     all_loader = Loader(scaled_research_df, scaled_research_df, batch_size, shuffle=False)
    train_iterator = Loader(X_train, y_train, batch_size, shuffle=False)
    val_iterator = Loader(X_test, y_test, batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    history_train = []
    history_val = []
    best_val_loss = float('+inf')
    show_progress_text =""
    for epoch in range(n_epochs):
        train_loss = model.run_epoch(train_iterator, optimiser, criterion, phase='train',
                                          device=device)  # , writer=writer)
        val_loss = model.run_epoch(val_iterator, None, criterion, phase='val',
                                        device=device)  # , writer=writer)

        history_train.append(train_loss)
        history_val.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_file)

        display.clear_output(wait=True)
        plt.figure()
        plt.plot(history_train, label='Train')
        plt.plot(history_val, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

        show_progress_text += f'Epoch: {epoch + 1:02} \n'
        show_progress_text += f'\tTrain Loss: {train_loss:.3f} \n'
        show_progress_text += f'\t Val. Loss: {val_loss:.3f} \n\n'
        print(show_progress_text)


    model.load_state_dict(torch.load(best_model_file))
    return model.run_epoch(val_iterator, None, criterion, phase='val', device=device) 