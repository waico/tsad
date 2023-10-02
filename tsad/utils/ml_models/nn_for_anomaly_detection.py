from tensorflow.keras.layers import Input, Conv2D, Conv1D, Dropout, Conv1DTranspose, ConvLSTM2D, Conv2DTranspose, TimeDistributed, LSTM, Dense, BatchNormalization, Activation, Flatten, Lambda, RepeatVector
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Model, losses
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import math
import numpy as np

class MSCRED:
    """
    MSCRED - Multi-Scale Convolutional Recurrent Encoder-Decoder first constructs multi-scale (resolution) signature matrices to characterize multiple levels of the system statuses across different time steps.  In particular, different levels of the system statuses are used to indicate the severity of different abnormal incidents. Subsequently, given the signature matrices, a convolutional encoder is employed to encode the inter-sensor (time series) correlations patterns and an attention based Convolutional Long-Short Term Memory (ConvLSTM) network is developed to capture the temporal patterns. Finally, with the feature maps which encode the inter-sensor correlations and temporal information, a convolutional decoder is used to reconstruct the signature matrices and the residual signature matrices are further utilized to detect and diagnose anomalies. The intuition is that MSCRED may not reconstruct the signature matrices well if it never observes similar system statuses before.

    Parameters
    ----------
    params : list
        A list containing configuration parameters for the MSCRED model.

    Attributes
    ----------
    model : Model
        The trained MSCRED model.

    Examples
    --------
    >>> from MSCRED import MSCRED
    >>> PARAMS = [sensor_n, scale_n, step_max]
    >>> model = MSCRED(PARAMS)
    >>> model.fit(X_train, Y_train, X_test, Y_test)
    >>> prediction = model.predict(test_data)
    """
    
    def __init__(self, params):
        self._Random(0)
        self.params = params

        input_size = (self.params[2], self.params[0], self.params[0], self.params[1])
        inputs = Input(input_size)#, batch_size=batch_size)

        if self.params[0] % 8 != 0:
            self.sensor_n_pad = (self.params[0] // 8) * 8 + 8
        else:
            self.sensor_n_pad = self.params[0]

        paddings = tf.constant([[0, 0], [0, 0], [0, self.sensor_n_pad-self.params[0]], 
                                [0, self.sensor_n_pad-self.params[0]], [0, 0]])
        inputs_pad = tf.pad(inputs, paddings)

        conv1 = TimeDistributed(Conv2D(filters = 32, kernel_size = 3, strides = 1, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv1'))(inputs_pad)

        conv2 = TimeDistributed(Conv2D(filters = 64, kernel_size = 3, strides = 2, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv2'))(conv1)

        conv3 = TimeDistributed(Conv2D(filters = 128, kernel_size = 2, strides = 2, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv3'))(conv2)

        conv4 = TimeDistributed(Conv2D(filters = 256, kernel_size = 2, strides = 2, 
                       kernel_initializer='glorot_uniform', padding='same', 
                       activation='selu', name = 'conv4'))(conv3)

        convLSTM1 = ConvLSTM2D(filters = 32, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM1")(conv1)
        convLSTM1_out = self.attention(convLSTM1, 1)

        convLSTM2 = ConvLSTM2D(filters = 64, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM2")(conv2)
        convLSTM2_out = self.attention(convLSTM2, 2)

        convLSTM3 = ConvLSTM2D(filters = 128, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM3")(conv3)
        convLSTM3_out = self.attention(convLSTM3, 4)

        convLSTM4 = ConvLSTM2D(filters = 256, kernel_size = 2, padding = 'same',
                               return_sequences = True, name="convLSTM4")(conv4)
        convLSTM4_out = self.attention(convLSTM4, 8)

        deconv4 = Conv2DTranspose(filters = 128, kernel_size = 2, strides = 2, 
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv4')(convLSTM4_out)
        deconv4_out = tf.concat([deconv4, convLSTM3_out], axis = 3, name = 'concat3')

        deconv3 = Conv2DTranspose(filters = 64, kernel_size = 2, strides = 2,
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv3')(deconv4_out)
        deconv3_out = tf.concat([deconv3, convLSTM2_out], axis = 3, name = 'concat2')

        deconv2 = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, 
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv2')(deconv3_out)
        deconv2_out = tf.concat([deconv2, convLSTM1_out], axis = 3, name = 'concat1')

        deconv1 = Conv2DTranspose(filters = self.params[1], kernel_size = 3, strides = 1, 
                                  kernel_initializer='glorot_uniform', padding = 'same', 
                                  activation='selu', name = 'deconv1')(deconv2_out)

        model = Model(inputs = inputs, outputs = deconv1[:, :self.params[0], :self.params[0], :])
        
        self.model = model
        
    def attention(self, outputs, koef):
        """
        Attention mechanism to weigh the importance of each step in the sequence.

        Parameters
        ----------
        outputs : tf.Tensor
            The output tensor from ConvLSTM layers.
        koef : int
            A coefficient to scale the attention mechanism.

        Returns
        -------
        tf.Tensor
            Weighted output tensor.
        """
        
        attention_w = []
        for k in range(self.params[2]):
            attention_w.append(tf.reduce_sum(tf.multiply(outputs[:,k], outputs[:,-1]), axis=(1,2,3))/self.params[2])
        attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w, axis=1)), [-1, 1, self.params[2]])
        outputs = tf.reshape(outputs, [-1, self.params[2], tf.reduce_prod(outputs.shape.as_list()[2:])])
        outputs = tf.matmul(attention_w, outputs)
        outputs = tf.reshape(outputs, 
                             [-1, math.ceil(self.sensor_n_pad/koef), math.ceil(self.sensor_n_pad/koef), 32*koef])
        return outputs
        
    def _Random(self, seed_value): 

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
        
    def _loss_fn(self, y_true, y_pred):
    
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def fit(self, X_train, Y_train, X_test, Y_test, batch_size=200, epochs=25):
        """
        Train the MSCRED model on the provided data.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training input data.
        Y_train : numpy.ndarray
            The training target data.
        X_test : numpy.ndarray
            The testing input data.
        Y_test : numpy.ndarray
            The testing target data.
        batch_size : int, optional
            The batch size for training, by default 200.
        epochs : int, optional
            The number of training epochs, by default 25.
        """

        self.model.compile(optimizer = Adam(learning_rate=1e-3),
                  loss = self._loss_fn,
                  )
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, 
                                      patience=6, min_lr=0.000001, 
                                      verbose = 1)
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                       validation_data = (X_test, Y_test),
                       callbacks=reduce_lr)

    def predict(self, data):
        """
        Generate predictions using the trained MSCRED model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return self.model.predict(data)
        
        
class Vanilla_LSTM:
    """
    LSTM-based neural network for anomaly detection using reconstruction error as an anomaly score.

    Parameters
    ----------
    params : list
        A list containing various parameters for configuring the LSTM model.

    Attributes
    ----------
    model : Sequential
        The trained LSTM model.

    Examples
    --------
    >>> from Vanilla_LSTM import Vanilla_LSTM
    >>> PARAMS = [N_STEPS, EPOCHS, BATCH_SIZE, VAL_SPLIT]
    >>> lstm_model = Vanilla_LSTM(PARAMS)
    >>> lstm_model.fit(train_data, train_labels)
    >>> predictions = lstm_model.predict(test_data)
    """
    
    def __init__(self, params):
        self.params = params
        
    def _Random(self, seed_value):

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
    
    def fit(self, X, y):
        """
        Train the LSTM model on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data for training the model.
        y : numpy.ndarray
            Target data for training the model.
        """

        self._Random(0)
        
        self.n_features = X.shape[2]
        model = Sequential()
        model.add(LSTM(100, 
                       activation='relu', 
                       return_sequences=True, 
                       input_shape=(self.params[0], self.n_features)))
        model.add(LSTM(100, 
                       activation='relu'))
        model.add(Dense(self.n_features))
        model.compile(optimizer='adam', 
                      loss='mae', 
                      metrics=["mse"])

        early_stopping = EarlyStopping(patience=10, 
                                       verbose=0)

        reduce_lr = ReduceLROnPlateau(factor=0.1, 
                                      patience=5, 
                                      min_lr=0.0001, 
                                      verbose=0)

        model.fit(X, y,
                  validation_split=self.params[3],
                  epochs=self.params[1],
                  batch_size=self.params[2],
                  verbose=0,
                  shuffle=False,
                  callbacks=[early_stopping, reduce_lr]
                  )
        
        self.model = model
    
    def predict(self, data):
        """
        Generate predictions using the trained LSTM model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return self.model.predict(data)

        
        
class Vanilla_AE:
    """
    Feed-forward neural network with autoencoder architecture for anomaly detection using reconstruction error as an anomaly score.

    Parameters
    ----------
    params : list
        List containing the following hyperparameters in order:
            - Number of neurons in the first encoder layer
            - Number of neurons in the bottleneck layer (latent representation)
            - Number of neurons in the first decoder layer
            - Learning rate for the optimizer
            - Batch size for training
            
    Attributes
    ----------
    model : tensorflow.keras.models.Model
        The autoencoder model.
                
    Examples
    -------
    >>> from Vanilla_AE import AutoEncoder
    >>> autoencoder = AutoEncoder(param=[5, 4, 2, 0.005, 32])
    >>> autoencoder.fit(train_data)
    >>> predictions = autoencoder.predict(test_data)
    """
    
    def __init__(self, params):
        self.param = params

        
    def _build_model(self):
        self._Random(0)

        input_dots = Input(shape=(self.shape,))
        x = Dense(self.param[0])(input_dots)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(self.param[1])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        bottleneck = Dense(self.param[2], activation='linear')(x)

        x = Dense(self.param[1])(bottleneck)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(self.param[0])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        out = Dense(self.shape, activation='linear')(x)

        model = Model(input_dots, out)
        model.compile(optimizer=Adam(self.param[3]), loss='mae', metrics=["mse"])
        self.model = model
        
        return model
        
    def _Random(self, seed_value):

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
    
    def fit(self, data, early_stopping=True, validation_split=0.2, epochs=40, verbose=0, shuffle=True):
        """
        Train the autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training.
        early_stopping : bool, optional
            Whether to use early stopping during training.
        validation_split : float, optional
            Fraction of the training data to be used as validation data.
        epochs : int, optional
            Number of training epochs.
        verbose : int, optional
            Verbosity mode (0 = silent, 1 = progress bar, 2 = current epoch and losses, 3 = each training iteration).
        shuffle : bool, optional
            Whether to shuffle the training data before each epoch.
        """

        self.shape = data.shape[1]
        self.model = self._build_model()
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(patience=3, verbose=0))
        self.model.fit(data, data,
                       validation_split=validation_split,
                       epochs=epochs,
                       batch_size=self.param[4],
                       verbose=verbose,
                       shuffle=shuffle,
                       callbacks=callbacks
                      )
    
    def predict(self, data):
        """
        Generate predictions using the trained autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for making predictions.

        Returns
        -------
        numpy.ndarray
            The reconstructed output predictions.
        """   
        
        return self.model.predict(data)
        
        
class Conv_AE: 
    """
    A reconstruction convolutional autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    No parameters are required for initializing the class.

    Attributes
    ----------
    model : Sequential
        The trained convolutional autoencoder model.

    Examples
    --------
    >>> from Conv_AE import Conv_AE
    >>> CAutoencoder = Conv_AE()
    >>> CAutoencoder.fit(train_data)
    >>> prediction = CAutoencoder.predict(test_data)
    """
    
    def __init__(self):
        self._Random(0)
        
    def _Random(self, seed_value): 

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
    
    def fit(self, data):
        """
        Train the convolutional autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training the autoencoder model.
        """

        model = Sequential(
            [
                Input(shape=(data.shape[1], data.shape[2])),
                Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        model.fit(
            data,
            data,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ],
        )
        
        self.model = model

    
    def predict(self, data):
        """
        Generate predictions using the trained convolutional autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return self.model.predict(data)
        
        
class LSTM_VAE:
    """
    A reconstruction LSTM variational autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    TenserFlow_backend : bool, optional
        Flag to specify whether to use TensorFlow backend (default is False).

    Attributes
    ----------
    None
        
    Examples
    -------
    >>> from LSTM_VAE import LSTM_VAE
    >>> model = LSTM_VAE()
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """
    
    def __init__(self, TenserFlow_backend=False):
        warnings.filterwarnings("ignore", category=UserWarning)
        self.TenserFlow_backend = TenserFlow_backend
        
    def _build_model(self, 
        input_dim, 
        timesteps, 
        batch_size, 
        intermediate_dim, 
        latent_dim,
        epsilon_std):
        
        self._Random(0)

        x = Input(shape=(timesteps, input_dim,))

        h = LSTM(intermediate_dim)(x)

        self.z_mean = Dense(latent_dim)(h)
        self.z_log_sigma = Dense(latent_dim)(h)
        
        if self.TenserFlow_backend:
            z = Lambda(self.sampling)([self.z_mean, self.z_log_sigma])
        else:
            z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_sigma])

        decoder_h = LSTM(intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(input_dim, return_sequences=True)

        h_decoded = RepeatVector(timesteps)(z)
        h_decoded = decoder_h(h_decoded)

        x_decoded_mean = decoder_mean(h_decoded)

        vae = Model(x, x_decoded_mean)

        encoder = Model(x, self.z_mean)

        decoder_input = Input(shape=(latent_dim,))

        _h_decoded = RepeatVector(timesteps)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)

        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        vae.compile(optimizer='rmsprop', loss=self.vae_loss)

        return vae, encoder, generator
        
    def _Random(self, seed_value):

        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        import random
        random.seed(seed_value)

        import numpy as np
        np.random.seed(seed_value)

        import tensorflow as tf
        tf.random.set_seed(seed_value)
        
    def sampling(self, args):
        """
        Sample from the latent space using the reparameterization trick.

        Parameters
        ----------
        args : list
            List of tensors [z_mean, z_log_sigma].

        Returns
        -------
        z : tensorflow.Tensor
            Sampled point in the latent space.
        """
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                  mean=0., stddev=self.epsilon_std)
        return z_mean + z_log_sigma * epsilon
    
    def vae_loss(self, x, x_decoded_mean):
        """
        Calculate the VAE loss.

        Parameters
        ----------
        x : tensorflow.Tensor
            Input data.
        x_decoded_mean : tensorflow.Tensor
            Decoded output data.

        Returns
        -------
        loss : tensorflow.Tensor
            VAE loss value.
        """
        mse = losses.MeanSquaredError()
        xent_loss = mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma))
        loss = xent_loss + kl_loss
        return loss
    
    def fit(self, data, epochs=20, validation_split=0.1, BATCH_SIZE = 1, early_stopping = True):
        """
        Train the LSTM variational autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training.
        epochs : int, optional
            Number of training epochs (default is 20).
        validation_split : float, optional
            Fraction of the training data to be used as validation data (default is 0.1).
        BATCH_SIZE : int, optional
            Batch size for training (default is 1).
        early_stopping : bool, optional
            Whether to use early stopping during training (default is True).
        """
        self.shape = data.shape
        self.input_dim = self.shape[-1] # 13
        self.timesteps = self.shape[1] # 3
        self.batch_size = BATCH_SIZE
        self.latent_dim = 100
        self.epsilon_std = 1.
        self.intermediate_dim = 32
        
        self.model, self.enc, self.gen = self._build_model(self.input_dim, 
        timesteps=self.timesteps, 
        batch_size=self.batch_size, 
        intermediate_dim=self.intermediate_dim,
        latent_dim=self.latent_dim,
        epsilon_std=self.epsilon_std)

        # self.model, self.enc, self.gen = create_lstm_vae(input_dim, 
        #     timesteps=timesteps, 
        #     batch_size=BATCH_SIZE, 
        #     intermediate_dim=32,
        #     latent_dim=100,
        #     epsilon_std=1.)
        
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0))
            
        self.model.fit(
            data,
            data,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=callbacks,
        )
    
    def predict(self, data):
        """
        Generate predictions using the trained LSTM variational autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for making predictions.

        Returns
        -------
        predictions : numpy.ndarray
            The reconstructed output predictions.
        """
        
        return self.model.predict(data)