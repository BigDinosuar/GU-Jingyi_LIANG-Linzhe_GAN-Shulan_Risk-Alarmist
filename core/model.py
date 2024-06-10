import os
import datetime as dt
import numpy as np
from numpy import newaxis
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from core.utils import Timer

class Model:
    """A class for building and training an LSTM model."""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        try:
            print('[Model] Loading model from file %s' % filepath)
            self.model = load_model(filepath)
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            
    
    def build_model(self, configs):
        timer = Timer()
        timer.start()
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
                self.model.add(BatchNormalization())
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print(f'[Model] {epochs} epochs, {batch_size} batch size')
        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.keras')
        # 设置EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss',     
            patience=10,           
            verbose=1,             
            mode='min',           
            restore_best_weights=True 
        )
        # 设置ModelCheckpoint
        model_checkpoint = ModelCheckpoint(
            'saved_models/30052024-215917-e5.keras',        
            monitor='val_loss',      
            verbose=1,                
            save_best_only=True,     
            save_freq='epoch'        
        )
        callbacks=[early_stopping, model_checkpoint]
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        self.model.save(save_fname)
        print(f'[Model] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print(f'[Model] {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')
        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.keras')
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit(data_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)
        print(f'[Model] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def predict_point_by_point(self, data):
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        return np.reshape(predicted, (predicted.size,))

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                prediction = self.model.predict(curr_frame[newaxis,:,:])[0,0]
                predicted.append(prediction)
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], prediction, axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        print('[Model] Predicting Full Sequence...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            prediction = self.model.predict(curr_frame[newaxis,:,:])[0,0]
            predicted.append(prediction)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], prediction, axis=0)
        return predicted
    
    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    def mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

