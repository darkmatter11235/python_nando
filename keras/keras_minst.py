from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

n_img_rows = 28
n_img_cols = 28
hid_dim = 512
out_dim = 10

n_train_samples = train_images.shape[0]
n_test_samples = test_images.shape[0]

train_images = train_images.reshape((n_train_samples, n_img_rows*n_img_cols))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((n_test_samples, n_img_rows*n_img_cols))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(hid_dim,activation='relu',input_shape=(n_img_rows*n_img_cols, )))
network.add(layers.Dense(out_dim, activation='softmax'))
 
network.compile(optimizer='rmsprop', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
n_epochs = 5
b_size = 128

print 'Training the network'
network.fit(train_images, train_labels, epochs=n_epochs, batch_size=b_size)

print 'Testing the network'
test_loss, test_accuracy = network.evaluate(test_images, test_labels)

print 'test_accuracy: %f, test_loss: %f'  %(test_accuracy, test_loss)
