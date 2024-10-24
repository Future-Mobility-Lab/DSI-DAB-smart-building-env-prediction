import tensorflow as tf
import tensorflow_probability as tfp
from abc import ABC, abstractmethod

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

class CNNLSTMModel(BaseModel):
    def build_model(self):
        # Define input layer
        inputs = tf.keras.Input(shape=(self.input_shape[1], self.input_shape[2]))
        
        # CNN Block
        x = inputs
        for i in range(self.config['cnn_layers']):
            filters = self.config['filters_1'] if i == 0 else self.config['filters_2']
            kernel_size = self.config['cnn_kernel_size_1'] if i == 0 else self.config['cnn_kernel_size_2']
            
            # Explicit layer creation and connection
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=self.config['activation'],
                name=f'conv1d_{i}'
            )(x)
            
            x = tf.keras.layers.MaxPooling1D(
                pool_size=2,
                padding='same',
                name=f'maxpool_{i}'
            )(x)
        
        # LSTM Block
        for i in range(self.config['lstm_layers']):
            return_sequences = i < (self.config['lstm_layers'] - 1)
            x = tf.keras.layers.LSTM(
                units=self.config['lstm_units'],
                activation=self.config['activation'],
                return_sequences=return_sequences,
                name=f'lstm_{i}'
            )(x)
        
        # Dense Block
        x = tf.keras.layers.Dense(
            units=self.config['dense_units'],
            activation='relu',
            name='dense_1'
        )(x)
        
        # Output Block
        if self.probabilistic:
            # Updated probabilistic output handling for TF 2.15+
            dist_params = tf.keras.layers.Dense(
                units=2 * self.config['prediction_length'],
                name='distribution_params'
            )(x)
            
            # Split the output into mu (mean) and sigma (standard deviation)
            mu, sigma = tf.split(dist_params, 2, axis=-1)
            sigma = tf.math.softplus(sigma) + 1e-6  # Ensure positive standard deviation
            
            # Create distribution using the Distribution Lambda layer
            outputs = tfp.layers.DistributionLambda(
                lambda params: tfp.distributions.Normal(loc=params[0], scale=params[1]),
                name='probabilistic_output'
            )([mu, sigma])
        else:
            outputs = tf.keras.layers.Dense(1, name='deterministic_output')(x)
        
        # Create and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CNNLSTMModel')
        return model

# Similar updates for CNNModel and LSTMModel classes
class CNNModel(BaseModel):
    def build_model(self):
        inputs = tf.keras.Input(shape=(self.input_shape[1], self.input_shape[2]))
        
        x = inputs
        for i in range(self.config['cnn_layers']):
            filters = self.config['filters_1'] if i == 0 else self.config['filters_2']
            kernel_size = self.config['cnn_kernel_size_1'] if i == 0 else self.config['cnn_kernel_size_2']
            
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=self.config['activation'],
                name=f'conv1d_{i}'
            )(x)
            
            x = tf.keras.layers.MaxPooling1D(
                pool_size=2,
                padding='same',
                name=f'maxpool_{i}'
            )(x)
        
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(self.config['dense_units'], activation='relu', name='dense_1')(x)
        
        if self.probabilistic:
            dist_params = tf.keras.layers.Dense(2 * self.config['prediction_length'], name='distribution_params')(x)
            mu, sigma = tf.split(dist_params, 2, axis=-1)
            sigma = tf.math.softplus(sigma) + 1e-6
            outputs = tfp.layers.DistributionLambda(
                lambda params: tfp.distributions.Normal(loc=params[0], scale=params[1]),
                name='probabilistic_output'
            )([mu, sigma])
        else:
            outputs = tf.keras.layers.Dense(1, name='deterministic_output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CNNModel')
        return model

class LSTMModel(BaseModel):
    def build_model(self):
        inputs = tf.keras.Input(shape=(self.input_shape[1], self.input_shape[2]))
        
        x = inputs
        for i in range(self.config['lstm_layers']):
            return_sequences = i < (self.config['lstm_layers'] - 1)
            x = tf.keras.layers.LSTM(
                units=self.config['lstm_units'],
                activation=self.config['activation'],
                return_sequences=return_sequences,
                name=f'lstm_{i}'
            )(x)
        
        x = tf.keras.layers.Dense(self.config['dense_units'], activation='relu', name='dense_1')(x)
        
        if self.probabilistic:
            dist_params = tf.keras.layers.Dense(2 * self.config['prediction_length'], name='distribution_params')(x)
            mu, sigma = tf.split(dist_params, 2, axis=-1)
            sigma = tf.math.softplus(sigma) + 1e-6
            outputs = tfp.layers.DistributionLambda(
                lambda params: tfp.distributions.Normal(loc=params[0], scale=params[1]),
                name='probabilistic_output'
            )([mu, sigma])
        else:
            outputs = tf.keras.layers.Dense(1, name='deterministic_output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='LSTMModel')
        return model