# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:54:26 2022

@author: agiokap
"""

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

model = tf.keras.models.Sequential([
    # 1st convolutional and max pooling layer + input layer
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = "relu", input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 2nd convolutional and max pooling layer
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 3rd convolutional and max pooling layer
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 4th convolutional and max pooling layer
    tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    # flatten layer
    tf.keras.layers.Flatten(),
    # 1st fully connected layerw
    tf.keras.layers.Dense(units = 512, activation = "relu"),
    # output layer
    tf.keras.layers.Dense(units = 1, activation = "sigmoid")])

model.compile(optimizer = RMSprop(learning_rate = +1e-3),
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

model.summary()

history = model.fit(training_generator,
                    steps_per_epoch = 100,
                    epochs = 100,
                    validation_data=validation_generator,
                    validation_steps = 50,
                    verbose = 2)