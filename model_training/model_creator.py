import tensorflow as tf
import tensorflow_probability as tfp
from abc import ABC, abstractmethod

# Base model class remains the same
class BaseModel(ABC):
    def __init__(self, input_shape, config, probabilistic=False):
        self.input_shape = input_shape
        self.config = config
        self.probabilistic = probabilistic
        tf.config.set_soft_device_placement(True)
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

# CNNModel remains Sequential
class CNNModel(BaseModel):
    def build_model(self):
        # Define input
        input_layer = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        
        # First CNN layer
        x = tf.keras.layers.Conv1D(
            filters=self.config['filters_1'],
            kernel_size=self.config['cnn_kernel_size_1'],
            padding='same',
            activation=self.config['activation']
        )(input_layer)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
        
        # Additional CNN layers
        for _ in range(self.config['cnn_layers'] - 1):
            x = tf.keras.layers.Conv1D(
                filters=self.config['filters_2'],
                kernel_size=self.config['cnn_kernel_size_2'],
                padding='same',
                activation=self.config['activation']
            )(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
        
        # Flatten and Dense layers
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.config['dense_units'], activation='relu')(x)
        
        # Output layer
        if self.probabilistic:
            x = tf.keras.layers.Dense(2 * self.config['prediction_length'])(x)
            outputs = tfp.layers.IndependentNormal(self.config['prediction_length'])(x)
        else:
            outputs = tf.keras.layers.Dense(1)(x)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=outputs)
        return model

# LSTMModel remains Sequential
class LSTMModel(BaseModel):
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2])),
            tf.keras.layers.LSTM(
                units=self.config['lstm_units'],
                activation=self.config['activation'],
                return_sequences=self.config['lstm_layers'] > 1
            )
        ])
        
        for i in range(1, self.config['lstm_layers']):
            model.add(tf.keras.layers.LSTM(
                units=self.config['lstm_units'],
                return_sequences=(i < self.config['lstm_layers'] - 1),
                activation=self.config['activation']
            ))
        
        model.add(tf.keras.layers.Dense(self.config['dense_units'], activation='relu'))
        
        if self.probabilistic:
            model.add(tf.keras.layers.Dense(2 * self.config['prediction_length']))
            model.add(tfp.layers.IndependentNormal(self.config['prediction_length']))
        else:
            model.add(tf.keras.layers.Dense(1))
        
        return model

# CNNLSTMModel using Functional API for proper layer connections
class CNNLSTMModel(BaseModel):
    def build_model(self):
        # Define input
        input_layer = tf.keras.layers.Input(shape=(self.input_shape[1], self.input_shape[2]))
        
        # CNN layers
        x = input_layer
        for i in range(self.config['cnn_layers']):
            filters = self.config['filters_1'] if i == 0 else self.config['filters_2']
            kernel_size = self.config['cnn_kernel_size_1'] if i == 0 else self.config['cnn_kernel_size_2']
            
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=self.config['activation']
            )(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
        
        # LSTM layers
        for i in range(self.config['lstm_layers']):
            return_sequences = i < self.config['lstm_layers'] - 1
            x = tf.keras.layers.LSTM(
                units=self.config['lstm_units'],
                activation=self.config['activation'],
                return_sequences=return_sequences
            )(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(self.config['dense_units'], activation='relu')(x)
        
        # Output layer
        if self.probabilistic:
            x = tf.keras.layers.Dense(2 * self.config['prediction_length'])(x)
            output_layer = tfp.layers.IndependentNormal(self.config['prediction_length'])(x)
        else:
            output_layer = tf.keras.layers.Dense(1)(x)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return model