# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.dept_classifier = layers.Dense(num_departments, activation="softmax")

    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)

        priority = self.priority_scorer(features)
        department = self.dept_classifier(features)
        return priority, department

# %% - Simulate data
vocabulary_size = 10000
num_tags = 100
num_samples = 1280
num_departments=4

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

# %%
model = CustomerTicketModel(num_departments=num_departments)
priority, department = model({
    'title': title_data,
    'text_body': text_body_data,
    'tags': tags_data
})

