# conda activate AudioClassifier
# https://www.vocitec.com/docs-tools/blog/sampling-rates-sample-depths-and-bit-rates-basic-audio-concepts
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
# Data from : https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing

import os
import matplotlib.pyplot as plt
import tensorflow as tf

import scipy.signal as sps
import librosa


# Avoid out of memory errors, by settting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    # print(gpu)
    tf.config.set_logical_device_configuration(
        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
    )


def load_wav_16k_mono(filename):
    # Go from sample_rate: 44100Hz to 16000Hz,
    new_rate = 16000
    wav, s = librosa.load(bytes.decode(filename.numpy()), sr=new_rate)
    return wav


CAPUCHIN_FILE = os.path.join("data", "Parsed_Capuchinbird_Clips", "XC3776-3.wav")
NOT_CAPUCHIN_FILE = os.path.join(
    "data", "Parsed_Not_Capuchinbird_Clips", "afternoon-birds-song-in-forest-0.wav"
)

POS = os.path.join("data", "Parsed_Capuchinbird_Clips")
NEG = os.path.join("data", "Parsed_Not_Capuchinbird_Clips")

# Create Tensorflow Datasets
pos = tf.data.Dataset.list_files(POS + "\*.wav")
neg = tf.data.Dataset.list_files(NEG + "\*.wav")

# Add labels, Combine all samples
positives = tf.data.Dataset.zip(
    (pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos))))
)
negatives = tf.data.Dataset.zip(
    (neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg))))
)
data = positives.concatenate(negatives)


# Preprocessing function
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]

    return (spectrogram, label)


# Visualizing spectrogram
# filepath, label = data.shuffle(buffer_size=10000).as_numpy_iterator().next()
# spectrogram, label = preprocess(filepath, label)
# plt.imshow(tf.transpose(spectrogram)[0])
# plt.show()


# Create Tensorflow Data Pipeline
data = data.map(
    lambda f, l: tf.py_function(
        func=preprocess, inp=[f, l], Tout=[tf.float32, tf.float32]
    )
)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train_data = data.take(36)
test_data = data.skip(36).take(15)


# Making the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(16, (3, 3), activation="relu", input_shape=(374, 129, 1)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:

        X = batch[0]  # the spectrogram
        y = batch[1]  # the label

        yhat = model(X, training=True)  # forward pass

        loss = binary_cross_loss(y, yhat)

    # calculate gradients
    grad = tape.gradient(loss, model.trainable_variables)

    # calculate updated weights and apply to model
    opt.apply_gradients(zip(grad, model.trainable_variables))

    return loss


def train(data, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print(f"\n Epoch {epoch}/{EPOCHS}")
        progbar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            train_step(batch)

            progbar.update(idx + 1)


EPOCHS = 4

# train(train_data, EPOCHS)

# model.save("audio_class_model.h5")


loaded_model = tf.keras.models.load_model("audio_class_model.h5")

test_input, y_true = test_data.as_numpy_iterator().next()

# making predictions : not that good result , maybe need to increase the prediction threshold
y_hat = model.predict([test_input])
y_hat = [1 if prediction > 0.5 else 0 for prediction in y_hat]
# print(y_hat)
# print(y_true)
