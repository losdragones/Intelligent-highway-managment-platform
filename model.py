# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:47:55 2022

@author: lenovo
"""

import os
import math
import numpy as np
import datetime as dt
import keras
from numpy import newaxis
from utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, Input
from keras.models import Sequential, load_model
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from attention import Attention
import matplotlib.pyplot as plt
from attention import attention

class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath, custom_objects={'attention': attention})

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
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'gru':
                self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            # if layer['type'] == 'attention':
            #     self.model.add(Attention(units=neurons))
        
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        
        print('[Model] Model Compiled')
        keras.utils.plot_model(self.model, show_shapes=True)
        timer.stop()
        
    def build_model_improve(self,configs):
        timer = Timer()
        timer.start()
        l = len(configs['data']['filepath']) + 2
        
        # input1 = Input(shape=(12,3))
        # gru1 = GRU(100, return_sequences=True)(input1)
        # dropout1 = Dropout(0.2)(gru1)
        # gru2 = GRU(100, return_sequences=True)(dropout1)
        # gru3 = GRU(100, return_sequences=True)(gru2)
        
        # input2 = Input(shape=(12,3))
        # gru4 = GRU(100, return_sequences=True)(input2)
        # dropout2 = Dropout(0.2)(gru4)
        # gru5 = GRU(100, return_sequences=True)(dropout2)
        # gru6 = GRU(100, return_sequences=True)(gru5)

        input1 = Input(shape=(12,l))
        gru1 = GRU(100, return_sequences=True)(input1)
        dropout1 = Dropout(0.2)(gru1)
        gru3 = GRU(100, return_sequences=True)(dropout1)
        
        input2 = Input(shape=(12,l))
        gru4 = GRU(100, return_sequences=True)(input2)
        dropout2 = Dropout(0.2)(gru4)
        gru6 = GRU(100, return_sequences=True)(dropout2)

        added = keras.layers.add([gru3, gru6])
        att = attention()(added)
        out = Dense(1, activation="linear")(att)
        
        self.model = keras.models.Model(inputs=[input1, input2], outputs=out)
        
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        
        print('[Model] Model Compiled')
        timer.stop()
        
    def build_model_improve_mul(self,configs,comp=True):
        timer = Timer()
        timer.start()
        l = len(configs['data']['filepath']) + 2
        
        input1 = Input(shape=(24,l))
        gru1 = GRU(100, return_sequences=True)(input1)
        dropout1 = Dropout(0.2)(gru1)
        gru3 = GRU(100, return_sequences=True)(dropout1)
        
        input2 = Input(shape=(24,l))
        gru4 = GRU(100, return_sequences=True)(input2)
        dropout2 = Dropout(0.2)(gru4)
        gru6 = GRU(100, return_sequences=True)(dropout2)

        added = keras.layers.add([gru3, gru6])
        att = attention()(added)
        out = Dense(12, activation="linear")(att)
        
        self.model = keras.models.Model(inputs=[input1, input2], outputs=out)
        
        if comp:
            self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
            print('[Model] Model Compiled')
       
        timer.stop()
        
    def train(self, x, y, file_num, inputstp, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        
        save_fname = os.path.join(save_dir,'f%s-i%s-%s.h5' % (str(file_num),str(inputstp),dt.datetime.now().strftime('%d%m%Y-%H%M%S')))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        
        # log_dir="./logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # cmd tensorboard --logdir file_name
        
        a = np.linspace(0.1, 1, 24)
        b = np.vstack((a for i in range(1748)))
        history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
            # sample_weight=b,
            # callbacks=callbacks
            # callbacks=[tensorboard_callback]
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        return history

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        history = self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )
        
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        # timer.stop()
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('???????????????/??????')
        # plt.xlabel('Epoch')
        # plt.legend(['Train_acc', 'Val_acc', 'Train_loss', 'Val_loss'])
        # plt.show()
        return history


    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
    
    def predict_sequence(self, data):
        print('[Model] Predicting Sequence...')
        predicted = self.model.predict(data)
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
