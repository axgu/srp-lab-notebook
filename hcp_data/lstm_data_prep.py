import numpy as np
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def setSeed(SEED = 66):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def determineTest():
    setSeed()
    index = np.arange(176)
    np.random.shuffle(index)
    testIndex = index[:76]
    return testIndex

def splitDict(dictionary, indexList):
    trainDict = {}
    testDict = {}
    for key, val in dictionary.items():
        testvalArr = []
        trainvalArr = []
        if key == "testretest":
            testvalArr = testvalArr
            for i in range(val.shape[0]):
                for p in range(val.shape[1]):
                    if p in indexList:
                        testvalArr.append(val[i][p])
                    else:
                        trainvalArr.append(val[i][p])
        else:
            for p in range(val.shape[0]):
                if p in indexList:
                        testvalArr.append(val[p])
                else:
                    trainvalArr.append(val[p])
        trainDict[key] = np.array(trainvalArr)
        testDict[key] = np.array(testvalArr)
    return trainDict, testDict

def shaping(dictionary, pad = -100.):
    X_arr = []
    y_arr = np.empty((0, 90))
    keylist = list(dictionary.keys())
    for key, val in dictionary.items():
        for i in range(val.shape[0]):
            normalized_seq = (val[i] - np.mean(val[i])) / np.std(val[i])
            X_arr.append(normalized_seq)
            clip = [key for j in range(min(val.shape[1], 90))]
            while len(clip) < 90:
                clip.append("")
            clip = np.array(clip).reshape((1, 90))
            y_arr = np.concatenate((y_arr, clip), axis=0)
    X_padded = paddingArr(np.array(X_arr), pad=pad)
    y_vector = vectorize_labels(y_arr, keylist, len(keylist))

    return X_padded, y_vector

def paddingArr(arr, max_len=90, num_features=300, pad = -100.):
    padded_arr = np.empty((arr.shape[0], max_len, num_features), dtype=float)
    for i, seq in enumerate(arr):
        rows = max(max_len - seq.shape[0], 0)
        pad_vals = [[pad for k in range(num_features)] for j in range(rows)]
        if rows > 0:
            newArr = np.concatenate((seq, np.array(pad_vals, dtype=float)), axis=0)
            padded_arr[i] = newArr
        else:
            padded_arr[i] = seq[:max_len, :]
    return padded_arr

def vectorize_labels(labels, keys, n_class):
    results = np.zeros((int(labels.shape[0]), int(labels.shape[1]), n_class))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] != '':
                indexNum = keys.index(labels[i][j])
                results[i][j][indexNum] = 1.
    return results

def numpy_prep(dictionary, pad = 0.):
    testIndex = determineTest()

    train, test = splitDict(dictionary, testIndex)
    X_train, y_train = shaping(train, pad = pad)
    X_test, y_test = shaping(test, pad = pad)
    return X_train, y_train, X_test, y_test

def prep(dictionary, pad=-1., batch_size=32):
    X_train, y_train, X_test, y_test = numpy_prep(dictionary, pad)
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    return train_loader, test_loader
