from ..base.task import Task, TaskResult


import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
from IPython import display


from .eda import HighLevelDatasetAnalysisResult


class DeepLeaningTimeSeriesForecastingResult(TaskResult):
    """
    A class representing the result of a deep learning time series forecasting task.
    """

    def show(self) -> None:
        """
        Displays the result of the task.
        """
        pass



class DeepLeaningTimeSeriesForecastingTask(Task):
    """ Multivariate Time Series Forecasting Task 
    based on SOTA deep learning forecasting algorithms. 
    """

    def __init__(self,
                 name: str | None = None,
                ):
        super().__init__(name) 

    def fit(self,
            dfs,
            result_base_eda: HighLevelDatasetAnalysisResult,
            model=None,
            optimiser = None,
            criterion = None,
            Loader = None,
            points_ahead = 1,
            n_epochs = 5, 
            len_seq = 10,  # Need to be fixed, need to be calculated from Result train_test_split
            batch_size = 128, 
            encod_decode_model = False, 
            random_state=None,
            shuffle=False,
            show_progress=True,
            show_figures=True,
            best_model_file='./best_model.pt',
            ):
        """
        Fit DeepLeaningTimeSeriesForecastingTask.

        Parameters
        ----------
        dfs : {{df*,ts*}, list of {df*,ts*}}
            df*,ts* are pd.core.series.Series or pd.core.frame.DataFrame data type.
            Initial data. The data should not contain np.nan at all, but have a constant
            and the same frequency of df.index and at the same time have no gaps. The problem with
            skipping solves splitting one df into list of dff.             
        
        model : object of torch.nn.Module class, default=models.SimpleLSTM()
            Used neural network model. 
        
        criterion : object of torch.nn class, default=nn.MSELoss()
            Error calculation criterion for optimization 
        
        optimiser : tuple = (torch.optim class ,default = torch.optim.Adam,
            dict  (dict of arguments without params models) , default=default)
            Example of optimiser : optimiser=(torch.optim.Adam,{'lr':0.001})
            Neural network optimization method and its parameters specified in
            documentation to torch.
            
        batch_size :  int, default=64
            Batch size (Number of samples over which the gradient is averaged)
        
        len_seq : int, default=10
            Window size (number of consecutive points in a row) on which
            the model really works. Essentially an analogue of order in autoregression.
        
        points_ahead : int, default=5
            Horizon forecasting 
        
        n_epochs :  int, default=100 
            Quantity epochs
            
            
        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
        
        show_progress : bool, default=True
            Whether or not to show learning progress with detail by epoch. 

        
        show_figures : bool, default=True
            Show or not show train and validation curve by epoch. 
        
        
        best_model_file : string, './best_model.pt'
            Path to the file where the best model weights will be stored
        
        Loader : class, default=ufesul.iterators.Loader.
            The type of loader that will be used as an iterator in the future,
            thanks to which, it is possible to hit the bachi .

        Returns 
        ----------
        y_pred : torch.tensor
            Tensor of predictions.
        result : DeepLeaningTimeSeriesForecastingResult
            Result of DeepLeaningTimeSeriesForecastingTask.

        """
        

        
        self.columns = result_base_eda.columns 

        if model is None:
            from ..utils.ml_models.deeplearning_regressors import SimpleLSTM
            model = SimpleLSTM(len(self.columns), len(self.columns), seed=random_state)
        self.model = model
        
        if optimiser is None:
            optimiser = torch.optim.Adam
            optimiser = optimiser(self.model.parameters())
        else:
            args = optimiser[1]
            args['params'] = self.model.parameters()
            optimiser = optimiser[0](**args)


        if Loader is None:
            from ..utils.iterators import Loader
        self.Loader = Loader

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if criterion is None:
            criterion = nn.MSELoss()

        self.points_ahead = points_ahead
        self.n_epochs = n_epochs
        self.len_seq = len_seq
        self.batch_size = batch_size
        self.encod_decode_model = encod_decode_model
        self.random_state = random_state
        self.shuffle = shuffle
        self.show_progress = show_progress
        self.show_figures = show_figures
        self.best_model_file = best_model_file
        

        if show_progress:
            show_progress_text = ""

        # -----------------------------------------------------------------------------------------
        #     Forming train_iterator and val_iteraror
        # -----------------------------------------------------------------------------------------

        X_train, X_test, y_train, y_test = dfs

        train_iterator = self.Loader(X_train, y_train, batch_size, shuffle=shuffle)
        val_iterator = self.Loader(X_test, y_test, batch_size, shuffle=shuffle)

        # -----------------------------------------------------------------------------------------
        #     Fit the model
        # -----------------------------------------------------------------------------------------

        history_train = []
        history_val = []
        best_val_loss = float('+inf')
        for epoch in range(n_epochs):
            train_loss = self.model.run_epoch(train_iterator, optimiser, criterion, phase='train',
                                              points_ahead=points_ahead, encod_decode_model=self.encod_decode_model,
                                              device=self.device)  # , writer=writer)
            val_loss = self.model.run_epoch(val_iterator, None, criterion, phase='val', points_ahead=points_ahead,
                                            encod_decode_model=self.encod_decode_model,
                                            device=self.device)  # , writer=writer)

            history_train.append(train_loss)
            history_val.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_file)

            if show_figures:
                display.clear_output(wait=True)
                plt.figure()
                plt.plot(history_train, label='Train')
                plt.plot(history_val, label='Val')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.legend()
                plt.show()

            if show_progress:
                show_progress_text = f'Epoch: {epoch + 1:02} \n' + \
                                     f'\tTrain Loss: {train_loss:.3f} \n' + \
                                     f'\t Val. Loss: {val_loss:.3f} \n\n' +  \
                                     show_progress_text
                print(show_progress_text)




        self.model.load_state_dict(torch.load(self.best_model_file))

        if show_progress:
            print("After choosing the best model:")
            try:
                test_iterator = self.Loader(X_test, y_test, len(X_test), shuffle=False)
                test_loss = self.model.run_epoch(test_iterator, None, criterion, phase='val',
                                                 encod_decode_model=self.encod_decode_model, device=self.device)
                print(f'Test Loss: {test_loss:.3f}')
            except:
                print('The entire X_test does not fit in memory, we test by averaging over batches')
                test_iterator = self.Loader(X_test, y_test, batch_size, shuffle=False)
                test_loss = []
                for epoch in range(n_epochs):
                    test_loss.append(self.model.run_epoch(test_iterator, None, criterion, phase='val',
                                                          encod_decode_model=self.encod_decode_model, device=self.device))
                print(f'Test Loss: {np.mean(test_loss):.3f}')

        all_data_iterator = self.Loader(X_train, y_train, self.batch_size, shuffle=False)
        y_pred = self.model.run_epoch( all_data_iterator,     
                                None, None, phase='forecast', points_ahead=self.points_ahead,
                                    device=self.device)

        result = DeepLeaningTimeSeriesForecastingResult()
        return y_pred, result


    # накосячил тут с прогнозом на одну точку вперед. Могут быть проблемы если ahead !=1
    def predict(self,
                dfs,
                result:DeepLeaningTimeSeriesForecastingResult,
                batch_size = None, 
                device=None,):
        
        """Predict by DeepLeaningTimeSeriesForecastingTask
        
        Parameters:
        ----------
        dfs : tuple
            Tuple of train and test data.
        result : DeepLeaningTimeSeriesForecastingResult
            Result of DeepLeaningTimeSeriesForecastingTask.
        batch_size : int, default=None
            Batch size (Number of samples over which the gradient is averaged)
        device : str, default=None
            Device to use for prediction.


        Returns:
        ----------
        y_pred : torch.tensor
            Tensor of predictions.
        result : DeepLeaningTimeSeriesForecastingResult
            Result of DeepLeaningTimeSeriesForecastingTask.
        """
        print(len(dfs[0]))
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.device  = device if device is not None else self.device
        
        X_train, X_test, y_train, y_test = dfs # тут нужен только X_train
        all_data_iterator = self.Loader(X_train, X_train, self.batch_size, shuffle=False) 
        
        y_pred = self.model.run_epoch( all_data_iterator,     
                                None, None, phase='forecast', points_ahead=self.points_ahead,
                                    device=self.device)

        return y_pred, result

