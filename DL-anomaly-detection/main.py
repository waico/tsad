from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


import src
import models

class LSTM_forecaster():
    # TODO требования к df, len(df.index.to_series().diff.dropna().unique())>1  , упорядоченность
    
    def __init__(
        self,
        preproc=None
                
    ):
        """

        
        preproc - способ преобработки в том числе шкалирования, по дефолту MinMaxScaler(), может быть свой но там должно быть в соответствии с sklearn
        sklearn.preprocessing.FunctionTransformer удобная функция 
 
        """
        if preproc is None:
            self.preproc = MinMaxScaler()

        
    # TODO где то пропадает df, найти в чем проблема    
    def fit(self,
            dfs,
            res_analys_alg,
            generate_res_func,
            model=None,
            criterion=None,
            optimiser=None,
            Loader=None,
            batch_size = 64,
            len_seq = 10,
            points_ahead = 5,
            n_epochs = 100,
            gap=0, 
            shag=1,
            intersection=True,
            test_size=0.2,
            train_size=None,
            random_state=None,
            shuffle=False,
            stratify=None,
            path_to_best_model = './best_model.pt',
            show_progress=True,
            show_figures=True
           ):

        """
    dfs -  обычный лист, так как если есть про должен быть либо качественно предообработан Жесткие требование по частоте дескритизации и пропускам. 
    or ts or df 
        """
        self.points_ahead = points_ahead
        self.len_seq = len_seq
        self.batch_size = batch_size
        dfs = dfs.copy()
         
            
                
        
        
        
        

# -----------------------------------------------------------------------------------------
#     Формирование train_iterator и val_iteraror
# -----------------------------------------------------------------------------------------
        if Loader is None:
            Loader = src.Loader

        if (type(dfs)== pd.core.series.Series) | (type(dfs) == pd.core.frame.DataFrame):
            df = dfs.copy() if type(dfs) == pd.core.frame.DataFrame else pd.DataFrame(dfs)
            self.columns = df.columns
            self.index = df.index
            new_df = pd.DataFrame(self.preproc.fit_transform(df),index=df.index,columns=df.columns)
            X_train, X_test, y_train, y_test= src.ts_train_test_split(new_df,
                                                                      len_seq,
                                                                      points_ahead=points_ahead,
                                                                      gap=gap,
                                                                      shag=shag,
                                                                      intersection=intersection,
                                                                      test_size=test_size,
                                                                      train_size=train_size,
                                                                      random_state=random_state,
                                                                      shuffle=False, # потому что потом нужно в основном итераторе
                                                                      stratify=stratify)

        elif type(dfs) == type(list()):
            # уже все pd.DataFrame
            _df = pd.concat(dfs,ignore_index=True)
            self.preproc.fit(_df)
            self.columns = _df.columns
            self.index  =  _df.index

            X_train, X_test, y_train, y_test = [],[],[],[]
            for df in dfs:
                if ((type(df) == pd.core.series.Series) | (type(df) == pd.core.frame.DataFrame))==False:
                    raise NameError('Type of dfs is unsupported')   
                
                new_df = pd.DataFrame(self.preproc.transform(df),index=df.index,columns=df.columns)
                _X_train, _X_test, _y_train, _y_test= src.ts_train_test_split(new_df,len_seq,
                                                                          points_ahead=points_ahead,
                                                                          gap=gap,
                                                                          shag=shag,
                                                                          intersection=intersection,
                                                                          test_size=test_size,
                                                                          train_size=train_size,
                                                                          random_state=random_state,
                                                                          shuffle=False,
                                                                          stratify=stratify)
                X_train += _X_train
                X_test += _X_test
                y_train += _y_train
                y_test += _y_test

        else:
            raise NameError('Type of dfs is unsupported')  
        

        train_iterator = Loader(X_train, y_train,batch_size,shuffle=shuffle)
        val_iterator   = Loader(X_test, y_test,batch_size,shuffle=shuffle)

# -----------------------------------------------------------------------------------------
#     Обучение моделей
# -----------------------------------------------------------------------------------------
            

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        
        if criterion is None:
            criterion = nn.MSELoss()
            
        if model is None:
            model = models.Model(len(self.columns),len(self.columns)).to(device)
            
            
        if optimiser is None:
            optimiser = torch.optim.Adam(model.parameters())
            
            
        history_train = []
        history_val = []
        best_val_loss = float('+inf')
        for epoch in range(n_epochs):    
            train_loss = src.run_epoch(model, train_iterator, optimiser, criterion, phase='train', points_ahead=points_ahead)#, writer=writer)
            val_loss = src.run_epoch(model, val_iterator, None, criterion, phase='val', points_ahead=points_ahead)#, writer=writer)

            history_train.append(train_loss)
            history_val.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path_to_best_model)
            
            if show_progress:
                print(f'Epoch: {epoch+1:02}')
                print(f'\tTrain Loss: {train_loss:.3f}')
                print(f'\t Val. Loss: {val_loss:.3f} ') 

                
        model.load_state_dict(torch.load(path_to_best_model))
        self.model = model
        print()
        
        if show_progress:
            try:
                test_iterator  = Loader(X_test, y_test,X_test.shape[0],shuffle=False)
                test_loss = src.run_epoch(model, test_iterator, None, criterion, phase='val')
                print(f'Test Loss: {test_loss:.3f}')
            except:
                print('Весь X_test не помещается в память, тестим усреднением по батчам')
                test_iterator  = Loader(X_test, y_test,batch_size,shuffle=False)
                test_loss = []
                for epoch in range(n_epochs):  
                    test_loss.append(src.run_epoch(model, test_iterator, None, criterion, phase='val'))
                print(f'Test Loss: {np.mean(test_loss):.3f}')
            
        if show_figures:
            plt.figure()
            plt.plot(history_train,label='Train')
            plt.plot(history_val,label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

# -----------------------------------------------------------------------------------------
#     Генерация остатков
# -----------------------------------------------------------------------------------------            
        # итератор для генерации остатков
        self.generate_res_func = generate_res_func
        self.res_analys_alg = res_analys_alg
        
        X      = X_train + X_test
        y_true = y_train + y_test
        all_data_iterator = Loader(X,y_true, batch_size,shuffle=False)
          
        y_pred = src.run_epoch(model, all_data_iterator, None, None, phase='forecast', points_ahead=points_ahead)       
        residuals = generate_res_func(y_pred,np.array(y_true))   
        
        point = 0 # мы иногда прогнозим на 10 точек вперед, ну интересует все равно на одну точку впреред 
        res_indices = [y_true[i].index[point] for i in range(len(y_true))]                                                
        df_residuals = pd.DataFrame(residuals[:,point,:],columns=self.columns,index=res_indices)
        anomaly_timestamps = self.res_analys_alg.fit_predict(df_residuals,show_figure=show_figures)
        
       
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   

    
    def forecast(self,df,points_ahead=None,Loader=None, show_figures=True):
        if Loader is None:
            Loader = src.Loader
        
        df  = df.copy()   
        points_ahead = points_ahead if points_ahead is not None else self.points_ahead
        len_seq = self.len_seq
        batch_size = self.batch_size         

        assert (type(df)== pd.core.series.Series) | (type(df) == pd.core.frame.DataFrame)
        df = df.copy() if type(df) == pd.core.frame.DataFrame else pd.DataFrame(df)
        df = df[-len_seq:]
        
        iterator = Loader(np.expand_dims(df.values,0),np.expand_dims(df.values,0), #ничего страшного, 'y' все равно не используется
                          batch_size,shuffle=False)

        y_pred = src.run_epoch(self.model, iterator, None, None, phase='forecast', points_ahead=points_ahead)[0]
        y_pred = self.preproc.inverse_transform(y_pred)
        
        t_last = np.datetime64(df.index[-1])
        delta_dime = np.timedelta64(df.index[-1] - df.index[-2]) 
        new_index = pd.to_datetime(t_last + np.arange(1,points_ahead+1)*delta_dime)
        y_pred = pd.DataFrame(y_pred,index=new_index,columns=df.columns)
        
        if show_figures:
            pd.concat([df,y_pred]).plot()
            plt.axvspan(t_last,y_pred.index[-1], alpha=0.2, color='green',label='forecast')
            plt.xlabel('Datetime')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        
        return y_pred



    
    # накосячил тут с прогнозом на одну точку вперед. Могут быть проблемы если ahead !=1
    def predict_anomaly(self,
                        dfs,
                        Loader=None,
                        gap=0, 
                        shag=1,
                        intersection=True,
                        train_size=None,
                        random_state=None,
                        shuffle=False,
                        stratify=None,
                        show_progress=True,
                        show_figures=True
           ):
        len_seq = self.len_seq
        batch_size = self.batch_size
        

        if Loader is None:
            Loader = src.Loader

        if (type(dfs)== pd.core.series.Series) | (type(dfs) == pd.core.frame.DataFrame):
            df = dfs.copy() if type(dfs) == pd.core.frame.DataFrame else pd.DataFrame(dfs)
            self.columns = df.columns
            self.index = df.index
            new_df = pd.DataFrame(self.preproc.fit_transform(df),index=df.index,columns=df.columns)
            X, y_true = src.ts_train_test_split(new_df,
                                          len_seq,
                                          points_ahead=1,
                                          gap=gap,
                                          shag=shag,
                                          intersection=intersection,
                                          test_size=0,
                                          random_state=random_state,
                                          shuffle=False, # потому что потом нужно в основном итераторе
                                          stratify=stratify)

        elif type(dfs) == type(list()):
            # уже все pd.DataFrame
            _df = pd.concat(dfs,ignore_index=True)
            self.preproc.fit(_df)
            self.columns = _df.columns
            self.index  =  _df.index

            X, y_true = [],[]
            for df in dfs:
                if ((type(df) == pd.core.series.Series) | (type(df) == pd.core.frame.DataFrame))==False:
                    raise NameError('Type of dfs is unsupported')   
                
                new_df = pd.DataFrame(self.preproc.transform(df),index=df.index,columns=df.columns)
                _X, _y = src.ts_train_test_split(new_df,len_seq,
                                                                          points_ahead=1,
                                                                          gap=gap,
                                                                          shag=shag,
                                                                          intersection=intersection,
                                                                          test_size=0,
                                                                          train_size=train_size,
                                                                          random_state=random_state,
                                                                          shuffle=False,
                                                                          stratify=stratify)
                X+= _X
                y_true += _y


        else:
            raise NameError('Type of dfs is unsupported')  
# -----------------------------------------------------------------------------------------
#     Генерация остатков
# -----------------------------------------------------------------------------------------            
        # итератор для генерации остатков
        generate_res_func = self.generate_res_func
        res_analys_alg = self.res_analys_alg
        
        all_data_iterator = Loader(X,y_true, batch_size,shuffle=False)
          
        y_pred = src.run_epoch(self.model, all_data_iterator, None, None, phase='forecast', points_ahead=1)     
        residuals = generate_res_func(y_pred,np.array(y_true))   
        
        point = 0 # мы иногда прогнозим на 10 точек вперед, ну интересует все равно на одну точку впреред 
        res_indices = [y_true[i].index[point] for i in range(len(y_true))]                                                
        df_residuals = pd.DataFrame(residuals[:,point,:],columns=self.columns,index=res_indices).sort_index()  
        anomaly_timestamps = self.res_analys_alg.fit_predict(df_residuals,show_figure=show_figures)
        

        