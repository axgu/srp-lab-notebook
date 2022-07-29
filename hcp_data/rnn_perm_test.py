import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

from lstm_data_prep import numpy_prep
from eval_model import find_lens
from lstm import test_model_lstm
from attention import test_model

def create_permutation_3d(X):
    np.random.seed()
    new_X = np.empty(X.shape)
    X_lens = find_lens(X)
    for batch in range(X.shape[0]):
        for feature in range(X.shape[-1]):
            randCol = X[batch, :X_lens[batch], feature]
            np.random.shuffle(randCol)
            new_X[batch, :X_lens[batch], feature] = randCol
            new_X[batch, X_lens[batch]:, feature] = np.array([-100.] * (X.shape[1] - X_lens[batch]))
    return new_X

def create_permutation_1d(X):
    np.random.seed()
    new_X = np.copy(X)
    np.random.shuffle(new_X)
    return new_X

def permutation_prep(dictionary, batch_size=32):
    _, _, X_test, y_test = numpy_prep(dictionary)

    permuted = create_permutation_3d(X_test)
    permuted_test_data = TensorDataset(torch.from_numpy(permuted).float(), torch.from_numpy(y_test).float())
    permuted_test_loader = DataLoader(permuted_test_data, shuffle=True, batch_size=batch_size)

    return permuted_test_loader

def iterateLSTM(model, loss, dictionary, percentile = 95, seq_len = 90, numSamples = 100):
    permutations = []
    for i in range(numSamples):
        permuted_test_loader = permutation_prep(dictionary)
        permutedAccuracy = test_model_lstm(model, permuted_test_loader, loss, seq_len)
        permutations.append(permutedAccuracy)
    permutations = np.array(permutations)
    plot = np.percentile(permutations, percentile, axis = 0)
    return plot

def iterateSeq(encoder, decoder, loss, dictionary, percentile = 95, seq_len = 90, numSamples = 100):
    permutations = []
    for i in range(numSamples):
        permuted_test_loader = permutation_prep(dictionary)
        
        permutedAccuracy, loss_val = test_model(encoder, decoder, permuted_test_loader, seq_len, loss)
        permutations.append(permutedAccuracy)
    permutations = np.array(permutations)
    # permutations = np.percentile(permutations, percentile, axis=0)
    return permutations

def generate_random_features(X_lens, num_batches, num_seq = 90, num_features = 300):
    X_random = []
    for i in range(num_batches):
        X_batch = np.random.normal(size=(X_lens[i], num_features))
        if X_lens[i] < num_seq:
            X_pad = np.array([[-100.]*300]*(num_seq - X_lens[i]))
            X_batch = np.append(X_batch, X_pad, axis=0)
        X_random.append(X_batch)
    X_random = np.array(X_random)
    return X_random

def test_random_features(encoder, decoder, loss, dictionary, seq_len = 90, num_samples = 3, batch_size = 32):
    _, _, X, y = numpy_prep(dictionary)
    X_lens = find_lens(X)
    random_features_acc = []
    for i in range(num_samples):
        X_random = generate_random_features(X_lens, X.shape[0])
        X_random_data = TensorDataset(torch.from_numpy(X_random).float(), torch.from_numpy(y).float())
        X_random_loader = DataLoader(X_random_data, shuffle=True, batch_size=batch_size)
        sample_acc, loss_val = test_model(encoder, decoder, X_random_loader, seq_len, loss)
        random_features_acc.append(sample_acc)
    return random_features_acc


def generate_random_labels(X_lens, num_batches, num_seq = 90, num_clips = 15):
    y_random = np.zeros((num_batches, num_seq, num_clips))
    for i in range(num_batches):
        indices = np.random.randint(15, size=X_lens[i])
        for j in range(X_lens[i]):
            y_random[i][j][indices[j]] = 1
    return y_random

def generate_random_column_labels(X_lens, num_batches, num_seq = 90, num_clips = 15):
    y_random = np.zeros((num_batches, num_seq, num_clips))
    indices = np.random.randint(15, size=num_batches)
    for i in range(num_batches):
        y_random[i, :X_lens[i], indices[i]] = np.array([1]*X_lens[i])
    return y_random

def test_random_labels(encoder, decoder, loss, dictionary, seq_len = 90, num_samples = 3, batch_size = 32):
    _, _, X, _ = numpy_prep(dictionary)
    X_lens = find_lens(X)
    labels_samples_acc = []
    for iteration in range(num_samples):
        y_random = generate_random_labels(X_lens, X.shape[0])
        random_labels = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y_random).float())
        random_labels_loader = DataLoader(random_labels, shuffle=True, batch_size=batch_size)
        sample_acc, loss_val = test_model(encoder, decoder, random_labels_loader, seq_len, loss)
        labels_samples_acc.append(sample_acc)
    return labels_samples_acc, loss_val

def test_random_column_labels(encoder, decoder, loss, dictionary, seq_len = 90, num_samples = 3, batch_size = 32):
    _, _, X, _ = numpy_prep(dictionary)
    X_lens = find_lens(X)
    labels_samples_acc = []
    for iteration in range(num_samples):
        y_random = generate_random_column_labels(X_lens, X.shape[0])
        random_labels = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y_random).float())
        random_labels_loader = DataLoader(random_labels, shuffle=True, batch_size=batch_size)
        sample_acc, loss_val = test_model(encoder, decoder, random_labels_loader, seq_len, loss)
        labels_samples_acc.append(sample_acc)
    return labels_samples_acc, loss_val
