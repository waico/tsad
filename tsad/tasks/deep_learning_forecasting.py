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

    def show(self) -> None:

        pass



class DeepLeaningTimeSeriesForecastingTask(Task):

    def __init__(self,
                 name: str | None = None,
                ):
        
        """Time Series Forecasting Task based on 
        SOTA deep learning forecasting algorithms.
            
        Parameters
        ----------
            Fit params

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

        Attributes
        ----------

        Return 
        ----------
        Forecast results in tensor form

        """
        
        
        super().__init__(name) 

  
        

    # -----------------------------------------------------------------------------------------
    #     Формирование сутевой части класса
    # -----------------------------------------------------------------------------------------


    def fit(self,
            dfs,
            result_base_eda: HighLevelDatasetAnalysisResult,
            model=None,
            optimiser = None,
            criterion = None,
            Loader = None,
            points_ahead = 1,
            n_epochs = 5, 
            len_seq = 10, 
            batch_size = 128, 
            encod_decode_model = False, 
            random_state=None,
            shuffle=False,
            show_progress=True,
            show_figures=True,
            best_model_file='./best_model.pt',
            ):
        """
        Метод fit
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
        #     Формирование train_iterator и val_iteraror
        # -----------------------------------------------------------------------------------------

        X_train, X_test, y_train, y_test = dfs

        train_iterator = self.Loader(X_train, y_train, batch_size, shuffle=shuffle)
        val_iterator = self.Loader(X_test, y_test, batch_size, shuffle=shuffle)

        # -----------------------------------------------------------------------------------------
        #     Обучение моделей
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
                print('Весь X_test не помещается в память, тестим усреднением по батчам')
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

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # накосячил тут с прогнозом на одну точку вперед. Могут быть проблемы если ahead !=1
    def predict(self,
                dfs,
                result:DeepLeaningTimeSeriesForecastingResult,
                points_ahead = None, 
                n_epochs = None, 
                len_seq = None, 
                batch_size = None, 
                encod_decode_model = None, 
                random_state = None, 
                shuffle = None, 
                show_progress = None, 
                show_figures = None, 
                best_model_file = None):

        """
        Метод predict 
        
        Parameters
        ----------
        см self.fit() dockstring
        
        
        Return
        ----------
        anomaly_timestamps : list of df.index.dtype
            Возвращает список временных меток аномалий                
        
        Attributes
        ----------
        
        """
        self.points_ahead = points_ahead if points_ahead is not None else self.points_ahead
        self.n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        self.len_seq = len_seq if len_seq is not None else self.len_seq
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.encod_decode_model = encod_decode_model if encod_decode_model is not None else self.encod_decode_model
        self.random_state = random_state if random_state is not None else self.random_state
        self.shuffle = shuffle if shuffle is not None else self.shuffle
        self.show_progress = show_progress if show_progress is not None else self.show_progress
        self.show_figures = show_figures if show_figures is not None else self.show_figures
        self.best_model_file = best_model_file if best_model_file is not None else self.best_model_file



        len_seq = self.len_seq
        # -----------------------------------------------------------------------------------------
        #     Генерация остатков
        # -----------------------------------------------------------------------------------------
        X_train, X_test, y_train, y_test = dfs # тут нужен только X_train
        all_data_iterator = self.Loader(X_train, X_train, self.batch_size, shuffle=False) 
        
        y_pred = self.model.run_epoch( all_data_iterator,     
                                None, None, phase='forecast', points_ahead=self.points_ahead,
                                    device=self.device)

        return y_pred, result

    def forecast(self, df, points_ahead=None, show_figures=True):
        """
        Прогнозирование временного ряда, в том числе векторного.
        
        Parameters
        ----------
        df : pd.core.series.Series or pd.core.frame.DataFrame data type
            Исходные данные. Данные не долнжны содержать np.nan вовсе, иметь постоянную 
            и одинковую частоту of df.index и при этом не иметь пропусков.         
                
        points_ahead : int, default=5
            Горизонт прогнозирования. 
               
        show_figures : bool, default=True
            Показывать или нет результаты решения задачии anomaly detection 
            и кривую трейна и валидации по эпохам. 
        
        
        Loader : class, default=iterators.Loader.
            Тип загрузчика, которую будет использовать как итератор в будущем, 
            благодаря которому, есть возможность бить на бачи.
        
                
        

        
        Attributes
        ----------
        
        """

        df = df.copy()
        points_ahead = points_ahead if points_ahead is not None else self.points_ahead
        len_seq = self.len_seq
        batch_size = self.batch_size

        assert (type(df) == pd.core.series.Series) | (type(df) == pd.core.frame.DataFrame)
        df = df.copy() if type(df) == pd.core.frame.DataFrame else pd.DataFrame(df)
        df = df[-len_seq:]
        assert not self._init_preproc
        preproc_values = self.preproc.transform(df)

        iterator = self.Loader(np.expand_dims(preproc_values, 0), np.expand_dims(preproc_values, 0),
                          # ничего страшного, 'y' все равно не используется
                          batch_size, shuffle=False)

        y_pred = self.model.run_epoch(iterator, None, None, phase='forecast', points_ahead=points_ahead, device=self.device)[
            0]
        y_pred = self.preproc.inverse_transform(y_pred)

        t_last = np.datetime64(df.index[-1])
        delta_dime = np.timedelta64(df.index[-1] - df.index[-2])
        new_index = pd.to_datetime(t_last + np.arange(1, points_ahead + 1) * delta_dime)
        y_pred = pd.DataFrame(y_pred, index=new_index, columns=df.columns)

        if show_figures:
            pd.concat([df, y_pred])[-3 * points_ahead:].plot()
            plt.axvspan(t_last, y_pred.index[-1], alpha=0.2, color='green', label='forecast')
            plt.xlabel('Datetime')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

        return y_pred

    def save(self, path='./pipeline.pcl'):
        """
        Method for saving pipeline.
        It may be required for example after training.
        CPU.
        
        Parameters
        ----------
            path : str
        Путь до файла, для сохранения пайплайна. 
        Пайлайн сохраняется в формате pickle
        """

        self.model.run_epoch(self.Loader(torch.zeros((1, self.len_seq, self.model.in_features), dtype=float),
                                        torch.zeros((1, self.len_seq, self.model.in_features), dtype=float),
                                        batch_size=1),
                             None, None, phase='forecast', points_ahead=1, device=self.device)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path='./pipeline.pcl'):
        """
        Method for loading pipeline.
        It may be required for example after training.
        
        Parameters
        ----------
            path : str
        Путь до сохраненного файла пайплайна. 
        Пайлайн должен быть в формате pickle
        """
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        self.__dict__.update(pipeline.__dict__)
