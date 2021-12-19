from typing import Sequence
from numpy import array

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_x = i + n_steps
        if end_x > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_x], sequence[end_x]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps = 3

X, y = split_sequence(seq, n_steps)

for i in range(len(X)):
    print(X[i], y[i])

