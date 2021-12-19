from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


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
n_features = 1
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], n_features))

LSTM_model = Sequential()
LSTM_model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
LSTM_model.add(Dense(1))
LSTM_model.compile(optimizer='adam', loss='mse')
LSTM_model.fit(X, y, epochs=200, verbose=0)

x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = LSTM_model.predict(x_input, verbose=0)
print(yhat)
