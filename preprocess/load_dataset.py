import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import losses
from keras.models import Sequential

from preprocess.data_preprocess import Preprocessor


def get_trunk(_X, idx, sample_len, rand_offset=False):
    '''Returns a trunk of the 1D array <_X>

    Params:
        _X: the concatenated audio samples
        idx: _X will be split in <sample_len> items. _X[idx]
        rand_offset: boolean to say whether or not we use an offset
    '''
    randint = np.random.randint(10000) if rand_offset is True else 0
    start_idx = (idx * sample_len + randint) % len(_X)
    end_idx = ((idx + 1) * sample_len + randint) % len(_X)
    if end_idx > start_idx:  # normal case
        return _X[start_idx: end_idx]
    else:
        return np.concatenate((_X[start_idx:], _X[:end_idx]))


def get_augmented_trunk(_X, idx, sample_len, added_samples=0):
    X = get_trunk(_X, idx, sample_len)

    # Add other audio of the same class to this sample
    for _ in range(added_samples):
        ridx = np.random.randint(len(_X))  # random index
        X = X + get_trunk(_X, ridx, sample_len)

    # One might add more processing (like adding noise)

    return X


def dataset_gen(dataset, is_train=True, batch_shape=(20, 16000), sample_augmentation=0):
    '''This generator is going to return training batchs of size <batch_shape>

    Params:
        is_train: True if you want the training generator
        batch_shape: a tupple (or list) consisting of 2 arguments, the number
            of samples per batchs and the number datapoints per samples
            (16000=1s)
        sample_augmentation: augment each audio sample by n other audio sample.
            Only works when <is_train> is True
    '''
    s_per_batch = batch_shape[0]
    s_len = batch_shape[1]

    X_cat = dataset['train_cat'] if is_train else dataset['test_cat']
    X_dog = dataset['train_dog'] if is_train else dataset['test_dog']

    # Random permutations (for X indexes)
    nbatch = int(max(len(X_cat), len(X_cat)) / s_len)
    perms = [list(enumerate([i] * nbatch)) for i in range(2)]
    perms = sum(perms, [])
    random.shuffle(perms)


    # Go through all the permutations
    y_batch = np.zeros(s_per_batch)
    X_batch = np.zeros(batch_shape)
    while len(perms) > s_per_batch:

        # Generate a batch
        for bidx in range(s_per_batch):
            perm, _y = perms.pop()  # Load the permutation
            y_batch[bidx] = _y

            # Select wether the sample is a cat or a dog
            _X = X_cat if _y == 0 else X_dog

            # Apply the permutation to the good set
            if is_train:
                X_batch[bidx] = get_augmented_trunk(
                    _X,
                    idx=perm,
                    sample_len=s_len,
                    added_samples=sample_augmentation)
            else:
                X_batch[bidx] = get_trunk(_X, perm, s_len)

        yield (X_batch.reshape(s_per_batch, s_len, 1),
               y_batch.reshape(-1, 1))

if __name__ == '__main__':
    path = os.path.join(Path(os.getcwd()).parent, 'data')
    # data = DataHolder()
    all_data = Preprocessor(data_path=path, is_mel=1, plot_spec=1)
    training_generator = dataset_gen(all_data.data.dataset, sample_augmentation=0)

    # Design model
    model = Sequential()
    # Choose the parameters to train the neural network

    model.add(layers.Dense(100, activation='relu', input_shape=(20, 16000)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])
    # Train model on dataset
    # model.compile()
    hist = model.fit_generator(generator=training_generator)

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label="training accuracy")
    plt.plot(epochs, val_acc, 'r', label="validation accuracy")
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.show()
