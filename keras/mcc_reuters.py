from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.datasets import reuters

import numpy as np
import matplotlib.pyplot as plt

num_words = 10000
(train_data, train_labels), (test_data, test_labels) =  reuters.load_data(
    num_words=num_words)

#print train_data.shape
#print train_data[0]
#print train_labels.shape
#print train_labels[0]

word_index = reuters.get_word_index()

reverse_word_index = dict([
    (value, key) for (key, value) in word_index.items()])

decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in 
                              train_data[0]])

#print decoded_newswire

def vectorize_sequences(sequences, num_words):
    results = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results

x_train = vectorize_sequences(train_data, num_words)
x_test = vectorize_sequences(test_data, num_words)

def one_hot_encode(labels, dimensions=46):
    results = np.zeros((len(labels), dimensions))
    for i, lval in enumerate(labels):
        results[i, lval] = 1
    return results

y_train = one_hot_encode(train_labels)
y_test = one_hot_encode(test_labels)


h1_dim = 64
h2_dim = 128
h3_dim = 128
out_dim = 46

network = models.Sequential()
network.add(layers.Dense(h1_dim, activation='relu', input_shape=(num_words,)))
network.add(layers.Dense(h2_dim, activation='relu'))
network.add(layers.Dense(h3_dim, activation='relu'))
network.add(layers.Dense(out_dim, activation='softmax'))

network.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.categorical_crossentropy,
                metrics=[metrics.categorical_accuracy])

n_total_train = train_data.shape[0]
val_pcent = 0.2
n_train = int(round(n_total_train*(1-val_pcent)))
part_x_train = x_train[:n_train]
part_y_train = y_train[:n_train]

x_val = x_train[n_train:]
y_val = y_train[n_train:]

cache = network.fit(part_x_train, part_y_train, 
            epochs = 20,
            batch_size = 512,
            validation_data = (x_val, y_val))

history = cache.history

val_loss = history['val_loss']
train_loss = history['loss']

x_epochs = range(1, len(val_loss)+1)
plt.plot(x_epochs, train_loss, 'bo', label='Training Loss')
plt.plot(x_epochs, val_loss, 'r', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
