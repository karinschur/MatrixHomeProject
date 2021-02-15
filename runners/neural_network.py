import os
import pickle
from pathlib import Path
from utils import utils
import numpy as np
from keras import layers
from keras import losses
from keras import models
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from preprocess.data_preprocess import Preprocessor


epochs = 70
load_from_data = False
file_suffix = 'sampling_noise'
path = os.path.join(Path(os.getcwd()).parent, 'data')
preprocessor = Preprocessor(over_sampling=True, noise_aug=True,
                            noise_aug_size=(2, 5), shift_aug=False,
                            loudness_aug=False, loud_aug_size=(2, 5))

if not load_from_data:
    train_x, train_y, test_x, test_y = preprocessor()
    with open(os.path.join(path, f'train_x_{file_suffix}.pkl'), 'wb') as f:
        pickle.dump(train_x, f)
    with open(os.path.join(path, f'train_y_{file_suffix}.pkl'), 'wb') as f:
        pickle.dump(train_y, f)
    with open(os.path.join(path, f'test_x_{file_suffix}.pkl'), 'wb') as f:
        pickle.dump(test_x, f)
    with open(os.path.join(path, f'test_y_{file_suffix}.pkl'), 'wb') as f:
        pickle.dump(test_y, f)
else:
    with open(os.path.join(path, f'train_x_{file_suffix}.pkl'), 'rb') as f:
        train_x = pickle.load(f)
    with open(os.path.join(path, f'train_y_{file_suffix}.pkl'), 'rb') as f:
        train_y = pickle.load(f)
    with open(os.path.join(path, f'test_x_{file_suffix}.pkl'), 'rb') as f:
        test_x = pickle.load(f)
    with open(os.path.join(path, f'test_y_{file_suffix}.pkl'), 'rb') as f:
        test_y = pickle.load(f)

print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}")
print(f"test_y shape: {test_y.shape}")

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

# Model
model = models.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape=train_x.shape))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Choose the parameters to train the neural network
checkpoint = [ModelCheckpoint(os.path.join(path, 'model.model'), monitor='val_acc', verbose=1,
                              save_best_only=True, mode='min', save_weights_only=False,
                              period=1)]

model.compile(optimizer='adam',
              loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(train_x, train_y,
                    validation_data=(val_x, val_y),
                    epochs=epochs,
                    shuffle=True,
                    verbose=1,
                    callbacks=checkpoint)


pred_y = model.predict(test_x)
pred_y_1d = np.argmax(pred_y, axis=1)
test_y_1d = np.nonzero(test_y)[1]
test_acc = sum([i == j for i, j in zip(pred_y_1d, test_y_1d)]) / len(test_y)
print(f'Test accuracy: {test_acc}')
print(f'Test set includes {sum(test_y_1d)} dogs and {len(test_y_1d) - sum(test_y_1d)} cats')
utils.plot_loss(epochs, history.history)
utils.plot_epoch_metrics(test_y_1d, pred_y_1d)
utils.plot_LS(pred_y[:,0], pred_y[:,1])