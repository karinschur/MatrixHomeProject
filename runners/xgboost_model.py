import os
import pickle
from pathlib import Path
import numpy as np
from typing import Dict
import xgboost as xgb
# from xgboost import XGBModel
from utils import utils
from utils import constants
from preprocess.data_preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


load_from_data = True
file_suffix = 'sampling_noise'
path = os.path.join(Path(os.getcwd()).parent, 'data/')
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

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15)

def loglikehood(labels, preds):
    #labels = train_data.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = (preds - labels)
    hess = preds * (1. - preds)
    return grad, hess


tree_nums = 500
xgb_params = {'learning_rate': 0.01, "n_estimators": tree_nums, "max_depth": 10}
# **xgb_params, objective='reg:logistic'


xgb_clf1 = xgb.XGBClassifier(**xgb_params)

xgb_clf1.fit(train_x, np.argmax(train_y, axis=1),
             eval_set=[(val_x, np.argmax(val_y, axis=1))],
             eval_metric='error', verbose=True, early_stopping_rounds=50)

pred_y = xgb_clf1.predict(test_x)
test_acc = sum([i == j for i, j in zip(pred_y, np.argmax(test_y, axis=1))]) / len(test_y)
print(test_acc)
# utils.plot_epoch_metric(epochs, acc, val_acc)

print(f'Number of useful features:{sum(xgb_clf1.feature_importances_>0)}')

utils.plot_epoch_metrics(np.argmax(test_y, axis=1), pred_y)
