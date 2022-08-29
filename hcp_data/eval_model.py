import numpy as np

def findIndex(val, arr):
    index = -1
    for x in range(arr.size):
        if val == arr[x]:
            index = x
            break
    return index

def accuracy(target, prob, correct, tot):
    for i in range(prob.shape[1]):
        correctCount = correct[i]
        totalCount = tot[i]
        for j in range(prob.shape[0]):
            if np.count_nonzero(target[j][i]) != 0:
                if findIndex(1., target[j][i]) == findIndex(1., prob[j][i]):
                    correctCount += 1
                totalCount += 1
        correct[i] = correctCount
        tot[i] = totalCount
    return correct, tot

def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    elapsed_mins = int(elapsed / 60)
    elapsed_secs = int(elapsed - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def find_lens(X):
    X_lens = []
    for batch in X:
        count = 0
        for time in batch:
            if time[0] == 0.:
                break
            else:
                count += 1
        X_lens.append(count)
    return X_lens
