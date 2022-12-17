# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# %%
inputs = keras.Input(shape=(3,), name="InputLayer")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# %% - Multi-input multi-output functional model
vocabulary_size = 10000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments, 
    activation="softmax", name="department")(features)

model = keras.Model(inputs=[title, text_body, tags],
    outputs=[priority, department])

# %% - Train the model

num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(
    optimizer="rmsprop",
    loss=["mean_squared_error", "categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]]
)
model.fit(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data],
    epochs=1
)
model.evaluate(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data]
)

priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data]
)
# %%
keras.utils.plot_model(model, show_shapes=True)