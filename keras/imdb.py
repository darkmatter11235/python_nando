import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras import losses
import matplotlib.pyplot as plt

num_words = 10000

(train_data, train_labels), (test_data, test_lables) = imdb.load_data(
    num_words=num_words)

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i-3, '?')
                           for i in train_data[0]])

#print decoded_review
def vectorized_sequences(seqs, num_words):
    results = np.zeros((len(seqs), num_words))
    for i, seq in enumerate(seqs):
        results[i,seq] = 1
    return results

x_train = vectorized_sequences(train_data, num_words)
#print train_data[0]
x_test = vectorized_sequences(test_data, num_words)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_lables).astype('float32')

h1_dim = 64
h2_dim = 128
h3_dim = 128
h4_dim = 128
out_dim = 1

model = models.Sequential()
model.add(layers.Dense(h1_dim, activation='relu', input_shape=(num_words, )))
model.add(layers.Dense(h2_dim, activation='relu'))
model.add(layers.Dense(h3_dim, activation='relu'))
model.add(layers.Dense(h4_dim, activation='relu'))
model.add(layers.Dense(out_dim, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              #loss=losses.binary_crossentropy,
              loss=losses.mse,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


cached_data = model.fit(partial_x_train, partial_y_train,
          epochs=20,
          batch_size=512,
          validation_data=(x_val, y_val))

history_dict = cached_data.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(val_loss) + 1 )
plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


#prediction = model.predict(x_test)

#print prediction
