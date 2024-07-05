
from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
from tensorflow import keras
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers import Input, Dropout, BatchNormalization, Dense

class Models(ABC):
    
    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

@dataclass
class CNN_LSTM_Model(Models):
    
    input_shape: Tuple[int, int]
    num_classes: int

    def define_model(self) -> Sequential:
        

        return Sequential(
            [
            Input(shape=self.input_shape),
            BatchNormalization(),

            #1D Convolutional layers
            Conv1D(32, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size = 3),
            Conv1D(64, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size = 3),
            Conv1D(128, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size = 3, padding='same'),
            Dropout(0.30),
            
            #LSTM layers
            LSTM(units = 128, return_sequences=True),
            SeqSelfAttention(attention_activation='tanh'),
            LSTM(units = 128, return_sequences=False),
            BatchNormalization(),
            Dropout(0.30),

            #Dense layers
            Dense(256, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.30),
            Dense(self.num_classes, activation='softmax')
            ]
        )

    def create_model(self) -> Sequential:
        
        model: Sequential = self.define_model()
        model.summary()
        return model