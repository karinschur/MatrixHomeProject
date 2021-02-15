from typing import List
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import librosa
import matplotlib.pyplot as plt


def librosa_read_wav_files(wav_files: List[str]):
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [librosa.load(f)[0] for f in wav_files]

# cats = 0, dogs = 1
def plot_epoch_metrics(y_test, y_pred):
    print('confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('MCC:')
    print(matthews_corrcoef(y_test, y_pred))


def plot_LS(LS_cats, LS_dogs):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].hist(LS_cats)
    axs[0].set_title('Cats likelihood score')
    axs[1].hist(LS_dogs)
    axs[1].set_title('Dogs likelihood score')
    plt.show()


def plot_loss(epochs, model_metrics):
    epoch_range = range(epochs)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].plot(epoch_range, model_metrics['loss'], "r--")
    axs[0].plot(epoch_range, model_metrics['val_loss'], "g--")
    axs[0].set_title('Loss')
    axs[1].plot(epoch_range, model_metrics['accuracy'], "r--")
    axs[1].plot(epoch_range, model_metrics['val_accuracy'], "g--")
    axs[1].set_title('Accuracy')
    plt.show()
