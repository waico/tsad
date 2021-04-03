from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



from . import src
from . import models
from . import generate_residuals
from . import stastics

class DL_AD():
    # TODO требования к df, len(df.index.to_series().diff.dropna().unique())>1  , упорядоченность
    
    def __init__(self,     
                 preproc=None,
                 generate_res_func=None,
                 res_analys_alg=None,
                 
                ):
        """
        Данный класс предназначен для прогнозирования временный рядов
        и обнаружение аномалий во временных рядах на основе алгоритмов
        глубокого обучения. 
        
        Преимущества:
        Недостатки:
        Read more in the :ref:`User Guide <svm_classification>`.
        
        Parameters
        ----------
        preproc : object, default=sklearn.preprocessing.MinMaxScaler()
            Данный объект для предобратки значений временного ряда.
            Требования к классу по методами атрибутам одинаковы с default.
        
        generate_res_func : func, default= generate_residuals.abs
            Функция генерация остатков имея y_pred, y_true. В default это
            абсолютная разница значений. Требования к функциям описаны в 
            generate_residuals.py. 
            
        res_analys_alg : object, default=stastics.hotteling().
            Объект поиска аномалий в остатках. В default это
            статистика Хоттелинга.Требования к классам описаны в 
            generate_residuals.py. 
            
                
        
        Attributes
        ----------
        
        
        Return 
        ----------
        object : object Объект этого класса DL_AD
        
            
        See Also ----------------------Next TODO- -------------
        --------
        SVC : About

        Notes
        -----
        The underlying C implementation uses a random number generator to
        select features when fitting the model. It is thus not uncommon
        to have slightly different results for the same input data. If
        that happens, try with a smaller ``tol`` parameter.

        The underlying implementation, liblinear, uses a sparse internal
        representation for the data that will incur a memory copy.

        Predict output may not match that of standalone liblinear in certain
        cases. See :ref:`differences from liblinear <liblinear_differences>`
        in the narrative documentation.

        References
        ----------
        
        Links to the papers 

        Examples
        --------
        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.pipeline import make_pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_features=4, random_state=0)
        >>> clf = make_pipeline(StandardScaler(),
        ...                     LinearSVC(random_state=0, tol=1e-5))
        >>> clf.fit(X, y)
        Pipeline(steps=[('standardscaler', StandardScaler()),
                        ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])

        >>> print(clf.named_steps['linearsvc'].coef_)
        [[0.141...   0.526... 0.679... 0.493...]]

        >>> print(clf.named_steps['linearsvc'].intercept_)
        [0.1693...]
        >>> print(clf.predict([[0, 0, 0, 0]]))
            
         
        """
        
        self.preproc = MinMaxScaler() if preproc is None else preproc
        self.generate_res_func = generate_residuals.abs if generate_res_func is None else generate_res_func
        self.res_analys_alg = stastics.hotteling() if res_analys_alg is None else res_analys_alg
        

        
    def fit(self,
            dfs,
            model=None,            
            criterion=None,
            optimiser=None,
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
            show_progress=True,
            show_figures=True,
            best_model_file = './best_model.pt',
            stratify=None,
            Loader=None,
           ):

        """
        Обучение модели как для задачи прогнозирования так и для задачи anomaly
        detection на имеющихся данных. 
        
        Parameters
        ----------
        dfs : {{df*,ts*}, list of {df*,ts*}}
            df*,ts* are pd.core.series.Seriesor or pd.core.frame.DataFrame data type.
            Исходные данные. Данные не долнжны содержать np.nan вовсе, иметь постоянную 
            и одинковую частоту of df.index и при этом не иметь пропусков. Проблему с 
            пропуском решают дробление одно df на list of df.             
        
        model : object of torch.nn.Module class, default=models.SimpleLSTM()
            Используемая модель нейронной сети. 
        
        criterion : object of torch.nn class, default=nn.MSELoss()
            Критерий подсчета ошибки для оптмизации. 
        
        optimiser : tuple = (torch.optim class ,default = torch.optim.Adam,
            dict  (dict of arguments without params models) , default=default)
            Example of optimiser : optimiser=(torch.optim.Adam,{'lr':0.001})
            Метод оптимизации нейронной сети и его параметры, указанные в 
            документации к torch.
            
        batch_size :  int, default=64
            Размер батча (Число сэмплов по которым усредняется градиент)
        
        len_seq : int, default=10
            Размер окна (количество последовательных точек ряда), на котором
            модель реально работает. По сути аналог порядка в авторегрессии. 
        
        points_ahead : int, default=5
            Горизонт прогнозирования. 
        
        n_epochs :  int, default=100 
            Количество эпох.
        
        ---------train_test_split----------------------------
        gap :  int, default=0
            Сколько точек между трейном и тестом. Условно говоря,
            если крайняя точка train а это t, то первая точка теста t + gap +1.
            Параметр создан, чтобы можно было прогнозировать одну точку через большой 
            дополнительный интервал времени. 
        
        shag :  int, default=1.
            Шаг генерации выборки. Если первая точка была t у 1-ого сэмпла трейна,
            то у 2-ого сэмла трейна она будет t + shag, если intersection=True, иначе 
            тоже самое но без пересечений значений ряда. 
        
        intersection :  bool, default=True
            Наличие значений ряда (одного момента времени) в различных сэмплах выборки. 
        
        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25. *
            *https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/model_selection/_split.py#L2076 
            Может быть 0, тогда вернет значения X,y
        
        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size. *
            *https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/model_selection/_split.py#L2076
        
        random_state : int, RandomState instance or None, default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.*
            *https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/model_selection/_split.py#L2076
            
        
        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None. *
        
        show_progress : bool, default=True
            Показывать или нет прогресс обучения с детализацией по эпохам. 

        
        show_figures : bool, default=True
            Показывать или нет результаты решения задачии anomaly detection 
            и кривую трейна и валидации по эпохам. 
        
        
        best_model_file : string, './best_model.pt'
            Путь до файла, где будет хранится лучшие веса модели
        
        Loader : class, default=src.Loader.
            Тип загрузчика, которую будет использовать как итератор в будущем, 
            благодаря которому, есть возможность бить на бачи.
        
                
        
        Attributes
        ----------

        """
        self.points_ahead = points_ahead
        self.len_seq = len_seq
        self.batch_size = batch_size
        dfs = dfs.copy()
        self.best_model_file = best_model_file
        
         
            
                
        
        
        
        

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
            model = models.SimpleLSTM(len(self.columns),len(self.columns)).to(device)
        model = model.to(device)
            
            
        if optimiser is None:
            optimiser = torch.optim.Adam
            optimiser = optimiser(model.parameters())
        else:
            args = optimiser[1]
            args['params']=model.parameters()
            optimiser = optimiser[0](**args)
            
            
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
                torch.save(model.state_dict(), self.best_model_file)
            
            if show_progress:
                print(f'Epoch: {epoch+1:02}')
                print(f'\tTrain Loss: {train_loss:.3f}')
                print(f'\t Val. Loss: {val_loss:.3f} ') 

                
        model.load_state_dict(torch.load(self.best_model_file))
        self.model = model
        print()
        
        if show_progress:
            try:
                test_iterator  = Loader(X_test, y_test, len(X_test),shuffle=False)
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
        X      = X_train + X_test
        y_true = y_train + y_test
        all_data_iterator = Loader(X,y_true, batch_size,shuffle=False)
          
        y_pred = src.run_epoch(model, all_data_iterator, None, None, phase='forecast', points_ahead=points_ahead)       
        residuals = self.generate_res_func(y_pred,np.array(y_true))   
        
        point = 0 # мы иногда прогнозим на 10 точек вперед, ну интересует все равно на одну точку впреред 
        res_indices = [y_true[i].index[point] for i in range(len(y_true))]                                                
        df_residuals = pd.DataFrame(residuals[:,point,:],columns=self.columns,index=res_indices)
        anomaly_timestamps = self.res_analys_alg.fit_predict(df_residuals,show_figure=show_figures)
        
       
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   

    
    def forecast(self,df,points_ahead=None,Loader=None, show_figures=True):
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
        
        
        Loader : class, default=src.Loader.
            Тип загрузчика, которую будет использовать как итератор в будущем, 
            благодаря которому, есть возможность бить на бачи.
        
                
        

        
        Attributes
        ----------
        
        """
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
        
        
        """
        Поиск аномалий в новом наборе данных
        
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
        all_data_iterator = Loader(X,y_true, batch_size,shuffle=False)
          
        y_pred = src.run_epoch(self.model, all_data_iterator, None, None, phase='forecast', points_ahead=1)     
        residuals = self.generate_res_func(y_pred,np.array(y_true))   
        
        point = 0 # мы иногда прогнозим на 10 точек вперед, ну интересует все равно на одну точку впреред 
        res_indices = [y_true[i].index[point] for i in range(len(y_true))]                                                
        df_residuals = pd.DataFrame(residuals[:,point,:],columns=self.columns,index=res_indices).sort_index()  
        anomaly_timestamps = self.res_analys_alg.fit_predict(df_residuals,show_figure=show_figures)
        
        return anomaly_timestamps
        

        