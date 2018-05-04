import sys, os
import _pickle as pk

# add directories in src/ to path
from core.data import get_data

sys.path.insert(0, '/home/zc/PycharmProjects/SpectralNet/src')

# define hyperparameters
params = {
    'dset': 'new_dataset',
    'val_set_fraction': 0.1,
    'precomputedKNNPath': '',
    'siam_batch_size': 128,

    'n_clusters': 4,
    'use_code_space': True,
    'affinity': 'siamese',
    'n_nbrs': 30,
    'scale_nbr': 10,
    'siam_k': 100,
    'siam_ne': 20,
    'spec_ne': 300,
    'siam_lr': 1e-3,
    'spec_lr': 5e-5,
    'siam_patience': 1,
    'spec_patience': 5,
    'siam_drop': 0.1,
    'spec_drop': 0.1,
    'batch_size': 2048,
    'siam_reg': 1e-2,
    'spec_reg': 7e-1,
    'siam_n': None,
    'siamese_tot_pairs': 400000,
    'arch': [
        {'type': 'relu', 'size': 512},
        {'type': 'relu', 'size': 256},
        {'type': 'relu', 'size': 128},
    ],
    'use_approx': True,
    'use_all_data': True,
}

# load dataset
old_x_train, x_train, x_test, old_y_train, y_train, y_test = pk.load(open('../data/protest/old_X_train.pk', 'rb')), \
                                                             pk.load(open('data/protest/X_train.pk', 'rb')), \
                                                             pk.load(open('data/protest/X_test.pk', 'rb')), \
                                                             pk.load(open('data/protest/old_Y_train.pk', 'rb')), \
                                                             pk.load(open('data/protest/Y_train.pk', 'rb')), \
                                                             pk.load(open('data/protest/Y_test.pk', 'rb'))

new_dataset_data = (x_train, x_test, y_train, y_test)

# preprocess dataset
data = get_data(params, new_dataset_data)

pk.dump(data, open('../new_data.pk', 'wb'))
