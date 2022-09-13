import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # nopep8
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
tf.config.list_physical_devices('GPU')

current_dir = "\\".join(os.path.realpath(__file__).split("\\")[:-1])
dataset_dir = os.path.join(current_dir, "Alzheimer_s Dataset")
train_dataset_dir = os.path.join(dataset_dir, 'train')
test_dataset_dir = os.path.join(dataset_dir, 'test')
IMG_SIZE = (208, 176)
SEED = 182
BATCH_SIZE = 16
EPOCHS = 10
COLOR_MODE = 'grayscale'
CHANNELS = 3 if COLOR_MODE == 'rgb' else 1
N_LABELS = len(os.listdir(train_dataset_dir))
DTYPE = tf.float64
MODEL_NAME = "Alzheimers_Model"


train_datagen = ImageDataGenerator(data_format='channels_last',
                                   validation_split=0.2,
                                   rescale=1.0 / 255.0,
                                   dtype=DTYPE)

test_datagen = ImageDataGenerator(data_format='channels_last',
                                  validation_split=0,
                                  rescale=1.0 / 255.0,
                                  dtype=DTYPE)

train_generator = train_datagen.flow_from_directory(directory=train_dataset_dir,
                                                    target_size=IMG_SIZE,
                                                    color_mode=COLOR_MODE,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset='training',
                                                    batch_size=BATCH_SIZE,
                                                    seed=SEED)

val_generator = train_datagen.flow_from_directory(directory=train_dataset_dir,
                                                  target_size=IMG_SIZE,
                                                  color_mode=COLOR_MODE,
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  subset='validation',
                                                  batch_size=BATCH_SIZE,
                                                  seed=SEED)

test_generator = test_datagen.flow_from_directory(directory=test_dataset_dir,
                                                  target_size=IMG_SIZE,
                                                  color_mode=COLOR_MODE,
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  subset='training',
                                                  batch_size=BATCH_SIZE,
                                                  seed=SEED)


@keras.utils.register_keras_serializable()
class ConvBlock(layers.Layer):
    def __init__(self, conv_filters, conv_kernel_size):
        super(ConvBlock, self).__init__()
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size

        self.conv1 = layers.Conv2D(self.conv_filters, self.conv_kernel_size, padding='same', activation=layers.LeakyReLU())
        self.conv2 = layers.Conv2D(self.conv_filters, self.conv_kernel_size, padding='same', activation=layers.LeakyReLU())
        self.bn = layers.BatchNormalization()
        self.maxpool = layers.MaxPooling2D(pool_size=2, strides=2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv_filters': self.conv_filters,
            'conv_kernel_size': self.conv_kernel_size
        })
        return config

    @tf.function()
    def call(self, tensor):
        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)
        tensor = self.bn(tensor)
        tensor = self.maxpool(tensor)
        return tensor

@keras.utils.register_keras_serializable()
class DenseBlock(layers.Layer):
    def __init__(self, units, dropout_rate=0):
        super(DenseBlock, self).__init__()
        self.units = units
        self.dropout_rate = dropout_rate

        self.dense = layers.Dense(self.units, activation=layers.LeakyReLU())
        self.bn = layers.BatchNormalization()
        if self.dropout_rate != 0:
            self.dropout = layers.Dropout(self.dropout_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

    @tf.function()
    def call(self, tensor):
        tensor = self.dense(tensor)
        tensor = self.bn(tensor)
        if self.dropout_rate != 0:
            tensor = self.dropout(tensor)
        return tensor


class Model(keras.Model):
    def __init__(self, n_labels=N_LABELS):
        super(Model, self).__init__()

        self.convblock1 = ConvBlock(32, 3)
        self.convblock2 = ConvBlock(64, 3)
        self.convblock3 = ConvBlock(128, 3)
        self.convblock4 = ConvBlock(256, 3)
        self.convblock5 = ConvBlock(256, 3)
        self.dropout = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense1 = DenseBlock(512, 0.5)
        self.dense2 = DenseBlock(256, 0.3)
        self.dense3 = DenseBlock(128, 0.2)
        self.outputs = layers.Dense(n_labels)

    @tf.function
    def call(self, tensor):
        tensor = self.convblock1(tensor)
        tensor = self.convblock2(tensor)
        tensor = self.convblock3(tensor)
        tensor = self.convblock4(tensor)
        tensor = self.convblock5(tensor)
        tensor = self.dropout(tensor)
        tensor = self.flatten(tensor)
        print(tensor.shape)
        tensor = self.dense1(tensor)
        tensor = self.dense2(tensor)
        tensor = self.dense3(tensor)
        tensor = self.outputs(tensor)
        return tensor

model = keras.Sequential(Model().layers)
model.build(input_shape=(None, *IMG_SIZE, CHANNELS))

# model = keras.Sequential([
#     layers.Input(shape=(*IMG_SIZE, CHANNELS)),
    
#     layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=2, strides=2),

#     layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=2, strides=2),

#     layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=2, strides=2),

#     layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=2, strides=2),

#     layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=2, strides=2),

#     layers.Flatten(),

#     layers.Dense(units=512, activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.Dropout(0.5),

#     layers.Dense(units=256, activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.Dropout(0.3),

#     layers.Dense(units=128, activation=layers.LeakyReLU()),
#     layers.BatchNormalization(),
#     layers.Dropout(0.2),

#     layers.Dense(units=N_LABELS)
    
# ])

model._name = MODEL_NAME
lr_scheduler = keras.optimizers.schedules.InverseTimeDecay(1e-4, decay_steps=len(train_generator), decay_rate=0.9)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
best_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(current_dir, f"{MODEL_NAME}.h5"),
                                                  monitor='val_acc',
                                                  save_best_only=True,
                                                  save_freq='epoch',
                                                  verbose=1)

model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),
              metrics=[keras.metrics.CategoricalAccuracy(name='acc')])

model.summary()

model.fit(train_generator,
          epochs=EPOCHS,
          verbose=1,
          workers=-1,
          use_multiprocessing=True,
          validation_data=val_generator,
          callbacks=[early_stopping, best_checkpoint])

results = model.evaluate(test_generator,
                         workers=-1,
                         use_multiprocessing=True,
                         verbose=1)