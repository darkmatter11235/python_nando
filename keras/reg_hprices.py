# Regression example modeling the house prices using 
# a very small datasets

from keras import layers, models, optimizers, losses, metrics
from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#print train_data.shape
#print "Before Norm"
#print train_data[0]
#print train_targets.shape
#print train_targets[0]

#column-wise normalization of the input features
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

#print "After Norm"
#print train_data[0]

def build_model():
    h1_dim = 64
    h2_dim = 64
    out_dim = 1
    model = models.Sequential()
    model.add(layers.Dense(h1_dim, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(h2_dim, activation='relu'))
    model.add(layers.Dense(out_dim))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#k-fold validation 
k = 4
# // operator for floor division i.e divide and floor
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_scores = []

for i in range(k):
    print "processing fold#%d" %(i)
    x_val = train_data[i*num_val_samples:(i+1)*num_val_samples]
    y_val = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    part_x_train = np.concatenate([train_data[:i*num_val_samples],
                                  train_data[(i+1)*num_val_samples:]], axis=0)
    part_y_train = np.concatenate([train_targets[:i*num_val_samples],
                                  train_targets[(i+1)*num_val_samples:]], axis=0)
    model = build_model()
    cache = model.fit(part_x_train, part_y_train, epochs=num_epochs, batch_size=1,
              verbose=0, validation_data=(x_val, y_val))
    #val_mse, val_mae = model.evaluate(x_val, y_val, verbose=0)
    all_mae_scores.append(cache.history['val_mean_absolute_error'])

avg_mae_scores = [
    np.mean([x[i] for x in all_mae_scores]) for i in range(num_epochs)]
plt.plot(range(1, len(avg_mae_scores)+1), avg_mae_scores)
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()
