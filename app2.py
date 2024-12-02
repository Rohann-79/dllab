from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

encoding_dim = 15
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(
    x_train, x_train,
    epochs=15,
    batch_size=256,
    validation_data=(x_test, x_test)
)

plt.figure(figsize=(20, 6))
encoded_img = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_img)

import random
i = random.randint(0, 10)

print("Original image")
ax = plt.subplot(3, 1, 1)
plt.imshow(x_test[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

print("Encoded image")
encoded_image = encoded_img[i].reshape(encoding_dim, 1)
ax = plt.subplot(3, 1, 2)
plt.imshow(encoded_image, aspect=0.05)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

print("Reconstructed image after decoding")
ax = plt.subplot(3, 1, 3)
plt.imshow(decoded_img[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()