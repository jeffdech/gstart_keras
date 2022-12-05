# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

# %%
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# %%

class Decoder:
    def __init__(self):
        word_index = reuters.get_word_index()
        self.idx = dict([(v, k) for (k, v) in word_index.items()])
    
    def decode(self, input):
        return " ".join([self.idx.get(i - 3, "?") for i in input])

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results
# %%
dec = Decoder()

# %%
print(dec.decode(train_data[10]))

# %%
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# %%
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"],
                      run_eagerly=True)

# %%
x_val = x_train[:1000]
px_train = x_train[1000:]
y_val = y_train[:1000]
py_train = y_train[1000:]

# %%
history = model.fit(
    px_train,
    py_train,
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

# %%
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"],
                      run_eagerly=True)

model.fit(
    px_train,
    py_train,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val)
)

results = model.evaluate(x_test, y_test)

# %%
