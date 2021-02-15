root_dir = '../data/cats_dogs/'
test_root_dir = '../data/cats_dogs/test'
train_root_dir = '../data/cats_dogs/train'
csv_path = '../data/cats_dogs/train_test_split.csv'
pickle_path = 'C:/Users/omera/karin/Matrix_CD/data/'
labels = ['train_cat', 'train_dog', 'test_cat', 'test_dog']

max_len = 39202
sampling_rate = 16000
duration = 2
hop_length = 347 * duration
fmin = 20
fmax = sampling_rate // 2
n_mels = 128
n_fft = n_mels * 20
samples = sampling_rate * duration
