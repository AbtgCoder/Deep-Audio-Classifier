import os
from itertools import groupby
import csv
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio


def load_mp3_16k_mono(filename):
    """
    Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio.

    Args:
        filename (string): path to audio file to be loaded
    """

    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2  # combine the channels

    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    wav = tfio.audio.resample(
        tensor, rate_in=sample_rate, rate_out=16000
    )  # resample to 16KHz

    return wav


def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


model = tf.keras.models.load_model("audio_class_model.h5")


results = {}
for file in os.listdir(os.path.join("data", "Forest Recordings")):
    FILEPATH = os.path.join("data", "Forest Recordings", file)

    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.preprocessing.timeseries_dataset_from_array(
        wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1
    )
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)

    yhat = model.predict(audio_slices)

    results[file] = yhat

class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]


postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum(
        [key for key, group in groupby(scores)]
    ).numpy()

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["recording", "capuchin_calls"])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
