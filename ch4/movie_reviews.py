# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# %%

class Decoder:
    def __init__(self):
            word_index = imdb.get_word_index()
            rev_index = dict([(val, key) for (key, val) in word_index.items()])
            self.idx = rev_index

    def decode(self, input):
        output = " ".join([self.idx.get(i - 3, "?") for i in input])
        return output

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results


# %%
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# %%
decoder = Decoder()
print(decoder.decode(train_data[0]))

# %%
x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype("float32")

x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype("float32")

# %%
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# %%
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# %%
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# %%
import matplotlib.pyplot as plt

history_dict = history.history
loss_vals = history_dict["loss"]
val_loss_vals = history_dict["val_loss"]
epochs = range(1, len(loss_vals) + 1)

plt.plot(epochs, loss_vals, "bo", label="Training loss")
plt.plot(epochs, val_loss_vals, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% - Retrain with 4 epochs
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=4,
    batch_size=512,
    validation_data=(x_val, y_val)
)
results = model.evaluate(x_test, y_test)

# %%
results

# %%
model.predict(x_test)