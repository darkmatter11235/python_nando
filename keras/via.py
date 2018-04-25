# Example for multi-class classification using keras 
# and IMDB sentiment analysis dataset

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import pandas

import numpy as np
import matplotlib.pyplot as plt

num_words = 10000
#dataset = pandas.read_csv('datasets/mldrc_via1_dataset.txt', header=None)
dataset = np.genfromtxt("datasets/mldrc_via1_dataset.txt", delimiter=",", skip_header=1)
dataset = np.asarray(dataset).astype('float32')
num_words = dataset.shape[0]
num_test = int(num_words*0.25)
num_train = num_words-num_test
X = dataset[:,:-6]
Y = dataset[:,-6:]
train_data = dataset[0:num_train,:-6]
train_labels = dataset[0:num_train,-6:]
print train_data[0:5]
print train_labels[0:5]
test_data = dataset[num_train:,:-6]
test_labels = dataset[num_train:,-6:]
#print dataset.shape

print train_data.shape
#print train_data[0]
print train_labels.shape
#print train_labels[0]

#y_train = train_labels
#y_test = test_labels
#mean = train_data.mean(axis=0)
#std = train_data.std(axis=0)

#train_data -= mean
#train_data /= std
#test_data -= mean
#test_data /= std

input_dim = train_data.shape[1]
print input_dim
h1_dim = 128
h2_dim = 128
h3_dim = 256
out_dim = 6
num_epochs = 10000
use_dropout = False
dropout_rate = 0.2
network = models.Sequential()
network.add(layers.Dense(h1_dim, activation='relu', input_shape=(input_dim,)))
if use_dropout == True:
    network.add(layers.Dropout(dropout_rate))
network.add(layers.Dense(h2_dim, activation='relu'))
if use_dropout == True:
    network.add(layers.Dropout(dropout_rate))
network.add(layers.Dense(h3_dim, activation='relu'))
if use_dropout == True:
    network.add(layers.Dropout(dropout_rate))
network.add(layers.Dense(out_dim, activation='sigmoid'))

#network.compile(optimizer=optimizers.RMSprop(lr=0.005),
#                loss=losses.binary_crossentropy,
#                metrics=[metrics.binary_accuracy])

network.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

#cache = network.fit(train_data, train_labels,
#            epochs = num_epochs,
#            batch_size = 512,
#            validation_data = (test_data, test_labels))

cache = network.fit(X, Y, validation_split=0.33,
            epochs = num_epochs,
            batch_size = 512)
history = cache.history

for key, value in history.iteritems():
    print key, value
val_loss = history['val_loss']
train_loss = history['loss']
train_accuracy = history['binary_accuracy']
val_accuracy = history['val_binary_accuracy']

#x_epochs = range(1, len(val_loss)+1)
#plt.plot(x_epochs, train_accuracy, 'bo', label='Training Loss')
#plt.plot(x_epochs, val_accuracy, 'r', label='Validation Loss')
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()
network.save('via1_drc.h5')
