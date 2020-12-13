import numpy as np
import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import *

np.random.seed(0)
tf.random.set_seed(0)

# Load images
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape images
img_size = 28
train_images = train_images.reshape((-1, img_size, img_size, 1))
test_images = test_images.reshape((-1, img_size, img_size, 1))

np.random.shuffle(train_labels)

# Build the model.
model = Sequential([
	Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
	Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
	MaxPooling2D((2, 2)),
  Dropout(0.25),
	Flatten(),
	Dense(100, activation='relu', kernel_initializer='he_uniform'),
	Dense(10, activation='softmax')
])

# Compile the model.
opt = SGD(lr=0.1, momentum=0.9)
model.compile(
  optimizer=opt,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=100,
  batch_size=128,
)

model.evaluate(
  test_images,
  to_categorical(test_labels)
)

model.save_weights('model.h5')

cdf = 0
total = 0
cdfs = []

y_pred = np.argmax(model.predict(train_images), axis=1)
for i in range(y_pred.shape[0]):
  total += 1
  if y_pred[i] == train_labels[i]:
    cdf += 1

  cdfs.append(cdf)

print(cdf, total)

from matplotlib import pyplot as plt
plt.plot(np.arange(y_pred.shape[0]))
plt.plot(cdfs)
plt.show()

inp = tf.Variable(train_images[0:1], dtype=tf.float32)

with tf.GradientTape() as tape:
    preds = model(inp)
    sum_preds = tf.reduce_sum(preds)
    print(preds)
grads = tape.gradient(sum_preds, inp)

plt.imshow(np.squeeze(inp.numpy()), cmap='gray')
plt.show()
plt.imshow(np.squeeze(grads.numpy()), cmap='gray')
plt.show()

