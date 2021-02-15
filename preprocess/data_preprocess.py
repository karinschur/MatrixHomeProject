import glob
from glob import glob
from typing import Tuple
import librosa
import librosa.display
import nlpaug.augmenter.audio as naa
import numpy as np
import pandas as pd
import scipy.io.wavfile as sci_wav
from imblearn.over_sampling import SMOTE
from keras import utils as keras_utils
from numpy import array as Array
from pandas import DataFrame as PandasDataFrame

from utils import constants
from utils import utils


DataType = Tuple[PandasDataFrame, Array, PandasDataFrame, Array]


class Preprocessor:

    def __init__(self, *, shift_aug: bool = False, noise_aug_size: Tuple[int, int], loud_aug_size: Tuple[float, int],
                 noise_aug: bool = False, over_sampling: bool = False, loudness_aug: bool = False):
        self.wav_rate = None
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data()
        self.shift_aug = shift_aug
        self.noise_aug = noise_aug
        self.over_sampling = over_sampling
        self.noise_aug_size = noise_aug_size
        self.loud_aug_size = noise_aug_size
        self.loudness_aug = loudness_aug

    def __call__(self) -> DataType:
        train_x, train_y, test_x, test_y = self.train_x, self.train_y, self.test_x, self.test_y

        if self.over_sampling:      # Balanced data using knn oversampling
            sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=100)
            # sm = KMeansSMOTE(sampling_strategy='auto', random_state=100, kmeans_estimator=10)
            train_x, train_y = sm.fit_resample(train_x, train_y)

        if self.shift_aug:
            aug_shift = naa.ShiftAug(self.wav_rate)
            augmented_data_shift = pd.DataFrame(aug_shift.augment(train_x))
            train_x = pd.concat([train_x, augmented_data_shift])
            train_y = np.tile(train_y, 2)

        if self.loudness_aug:
            aug_loud = naa.LoudnessAug(factor=self.loud_aug_size)
            augmented_data_loud = aug_loud.augment(train_x)
            train_x = pd.concat([train_x, augmented_data_loud])
            train_y = np.tile(train_y, 2)

        if self.noise_aug:
            aug_noise = naa.LoudnessAug(factor=self.noise_aug_size)
            augmented_data_noise = aug_noise.augment(train_x)
            train_x = pd.concat([train_x, augmented_data_noise])
            train_y = np.tile(train_y, 2)

        train_x, test_x = [self.extract_features(df, self.wav_rate) for df in [train_x, test_x]]
        train_y, test_y = [keras_utils.to_categorical(arr) for arr in [train_y, test_y]]

        return train_x, train_y, test_x, test_y

    def load_data(self) -> DataType:
        dogs_train = glob(constants.train_root_dir + "/dog/*")
        dogs_test = glob(constants.test_root_dir + "/dog/*")
        cats_train = glob(constants.train_root_dir + "/cat/*")
        cats_test = glob(constants.test_root_dir + "/cat/*")
        self.wav_rate = sci_wav.read(dogs_train[0])[0]
        train_x = (pd.DataFrame(
            {i: pd.Series(rec) for i, rec in
             enumerate(utils.librosa_read_wav_files(cats_train + dogs_train))})
                   .fillna(0).transpose())
        train_y = np.concatenate((np.zeros(len(cats_train)), np.ones(len(dogs_train))))
        test_x = (pd.DataFrame(
            {i: pd.Series(rec) for i, rec in
             enumerate(utils.librosa_read_wav_files(cats_test + dogs_test))})
                  .fillna(0).transpose())
        test_y = np.concatenate((np.zeros(len(cats_test)), np.ones(len(dogs_test))))

        return train_x, train_y, test_x, test_y

    @staticmethod
    def extract_features(audio_samples: PandasDataFrame, sample_rate: float):
        extracted_features = np.empty((0, constants.max_len + 1))
        audio_samples = audio_samples.to_numpy()

        for sample in audio_samples:
            # calculate the zero-crossing feature
            zero_cross_feat = librosa.feature.zero_crossing_rate(sample).mean()

            # calculate the mfccs features
            mfcc_2D = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40)
            mfcc_1D = mfcc_2D.flatten()
            mfccsscaled = np.pad(mfcc_1D, (0, constants.max_len - len(mfcc_1D)), 'constant')

            # add zero crossing feature to the feature list
            mfccsscaled = np.append(mfccsscaled, zero_cross_feat)
            mfccsscaled = mfccsscaled.reshape(1, constants.max_len + 1)

            extracted_features = np.vstack((extracted_features, mfccsscaled))

        # return the extracted features
        return extracted_features

    @staticmethod
    def audio_to_melspectrogram(audio: Array):
        spectrogram = librosa.feature.melspectrogram(audio,
                                                     sr=constants.sampling_rate,
                                                     n_mels=constants.n_mels,
                                                     hop_length=constants.hop_length,
                                                     n_fft=constants.n_fft,
                                                     fmin=constants.fmin,
                                                     fmax=constants.fmax)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram
