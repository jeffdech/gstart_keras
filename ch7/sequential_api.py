# %%
from tensorflow import keras
from tensorflow.keras import layers

# %%
model = keras.Sequential([
    layers.Dense(64, activation="relu", name="DenseLayer1"),
    layers.Dense(10, activation="softmax", name="OutputDenseLayer")
])

model.build(input_shape=(None, 3))

model.summary()

# %%
model = keras.Sequential()
model.add(keras.Input(shape=(3, )))
model.add(layers.Dense(64, activation="relu"))
model.summary()
model.add(layers.Dense(10, activation="softmax"))
model.summary()