# %%
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# %%
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# %%
# prep the data
train_images = train_images.reshape((-1, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((-1, 28 * 28))
test_images = test_images.astype("float32") / 255

# %%
# fit the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# %% - perform some predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0].argmax(), test_labels[0]

# %% - evaluate performance
test_loss, test_acc = model.evaluate(test_images, test_labels)

# %% - view test image
import matplotlib.pyplot as plt

digit = train_images[4].reshape((28, 28))
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()