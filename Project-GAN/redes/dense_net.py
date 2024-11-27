import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, latent_dim, ts_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ts_dim = ts_dim

        self.noise_to_latent = tf.keras.Sequential([
            tf.keras.layers.Dense(50, input_shape=(self.latent_dim,)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(200),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(2 * self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(2 * self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(self.ts_dim),
            # layers.Activation('tanh') # Descomenta si quieres usar la activación tanh al final
        ])

    def call(self, input_data):
        x = self.noise_to_latent(input_data)
        return x  # Equivalente a x[:, None, :] en PyTorch


class Discriminator(tf.keras.Model):
    def __init__(self, ts_dim):
        super(Discriminator, self).__init__()

        self.ts_dim = ts_dim

        self.features_to_score = tf.keras.Sequential([
            tf.keras.layers.Dense(2 * self.ts_dim, input_shape=(self.ts_dim,)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4 * self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(5 * self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(5 * self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(6 * self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(2 * self.ts_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(self.ts_dim)
        ])

    def call(self, input_data):
        x = self.features_to_score(input_data)
        return x
