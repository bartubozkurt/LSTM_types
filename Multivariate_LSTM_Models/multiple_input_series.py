from numpy import array
from numpy import hstack
from numpy.lib.shape_base import split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_x = i + n_steps
        if end_x > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_x, :-1], sequence[end_x-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))
# print(dataset)
n_steps = 3

X, y = split_sequence(dataset, n_steps)
#print(X.shape, y.shape)

#for i in range(len(X)):
#   print(X[i], y[i])
n_features = X.shape[2]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(X,y, epochs=200, verbose=0)

x_input = array([[80,85],[90,95],[100,105]])
x_input = x_input.reshape((1,n_steps,n_features))
yhat = model.predict(x_input,verbose=0)
print(yhat)

