# Base model class
class BaseModel(ABC):
    def __init__(self, input_shape, config, probabilistic=False):
        self.input_shape = input_shape
        self.config = config
        self.probabilistic = probabilistic
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        pass

# Specific model implementations
class CNNModel(BaseModel):
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=self.config['filters_1'], kernel_size=self.config['cnn_kernel_size_1'], padding='same',
                                   activation=self.config['activation'], input_shape=(self.input_shape[1], self.input_shape[2])),
            tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')
        ])
        for _ in range(self.config['cnn_layers'] - 1):
            model.add(tf.keras.layers.Conv1D(filters=self.config['filters_2'], kernel_size=self.config['cnn_kernel_size_2'],
                                             padding='same', activation=self.config['activation']))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding='same'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.config['dense_units'], activation='relu'))
        if self.probabilistic:
            model.add(tf.keras.layers.Dense(2 * self.config['prediction_length']))
            model.add(tfp.layers.IndependentNormal(self.config['prediction_length']))
        else:
            model.add(tf.keras.layers.Dense(1))
        return model

class LSTMModel(BaseModel):
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=self.config['lstm_units'], activation=self.config['activation'], return_sequences=self.config['lstm_layers'] > 1, input_shape=(self.input_shape[1], self.input_shape[2]))
        ])
        for i in range(1, self.config['lstm_layers']):
            model.add(tf.keras.layers.LSTM(units=self.config['lstm_units'], return_sequences=(i < self.config['lstm_layers']-1), activation=self.config['activation']))

        model.add(tf.keras.layers.Dense(self.config['dense_units'], activation='relu'))
        if self.probabilistic:
            model.add(tf.keras.layers.Dense(2 * self.config['prediction_length']))
            model.add(tfp.layers.IndependentNormal(self.config['prediction_length']))
        else:
            model.add(tf.keras.layers.Dense(1))
        return model

class CNNLSTMModel(BaseModel):
    def build_model(self):
        # Assuming the last dimension of self.input_shape is features which can be 1 or more (1 for univariate, more for multivariate)
        model = tf.keras.Sequential([
            # Add a Reshape layer to ensure the input is 4D (batch, timesteps, features, 1)
            tf.keras.layers.Reshape((self.input_shape[1], self.input_shape[2], 1), input_shape=(self.input_shape[1], self.input_shape[2])),

            # Apply TimeDistributed Conv1D to handle each timestep as a separate batch
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv1D(
                    filters=self.config['filters_1'],
                    kernel_size=self.config['cnn_kernel_size_1'],
                    padding='same',
                    activation=self.config['activation']
                )
            ),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),

            # LSTM layer to process the sequence of pooled feature vectors
            tf.keras.layers.LSTM(units=self.config['lstm_units'], activation=self.config['activation']),

            # Output layers
            tf.keras.layers.Dense(units=self.config['dense_units'], activation='relu')
        ])
        if self.probabilistic:
            model.add(tf.keras.layers.Dense(2 * self.config['prediction_length']))
            model.add(tfp.layers.IndependentNormal(self.config['prediction_length']))
        else:
            model.add(tf.keras.layers.Dense(1))
        return model