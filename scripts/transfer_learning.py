# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:40:21 2022

@author: agiokap
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

# step 1: initialize base model (InceptionV3)
def create_pretrained_model(input_shape, include_top = False, weights = "imagenet", train_weights = False):
    '''
    parameters
    ----------
    input_shape : tuple
        the input of the images that will be fed into the pretrained network.
    include_top : boolean, optional
        if true, the densely connected classifier of the pretrained network will
        be included. this part of the network is highly specialized to the initial
        problem that the pretrained model was trained. The default is False.
    weights : string, optional
        either "imagenet" (default), which downloads the pretrained wheights and
        applies them to the model's selected part, or a direct path to the locally
        stored wheights' file in the system. The default is "imagenet".
    train_weights: boolean, optional
        if true, the loaded weights will be considered trainable and will get updated
        during training phase. The default is False.
        
    returns
    -------
    '''
    
    pretrained_model = InceptionV3(include_top = include_top,
                                   weights = weights,
                                   input_shape = input_shape)
    
    pretrained_model.trainable = train_weights
    
    return pretrained_model

pretrained_model = create_pretrained_model(input_shape = (150, 150, 3),
                                           include_top = False,
                                           weights = "imagenet")

# get the shape of the output from convolutional_base
# that the densely connected classifier will have to follow.
# in this case, where convolutional_base is derived from InceptionV3,
# the layer, called "mixed7", is a good generalized layer, where we
# can connect our classifier, specialized in the data at hand.
def get_last_output(pretrained_model, last_layer):
    '''
    parameters
    ----------
    pretrained_model : keras.engine.functional.Functional
        the convolutional base, on top of which densely connected layers will
        be connected
    last_layer : string
        name of the last layer to keep from pretrained_model
        
    returns
    -------
    last_output : keras.engine.keras_tensor.KerasTensor
        output of last_layer
    last_shape : tuple
        output of last_layer's shape
    '''
    
    last_output = pretrained_model.get_layer(last_layer).output
    last_shape = tuple(last_output.shape)[1:]
    
    return last_output, last_shape

last_output, last_shape = get_last_output(pretrained_model, last_layer)

def create_model(pretrained_model, last_output, compiled = True, learning_rate = +1e-4):
    '''
    parameters
    ----------
    pretrained_model : keras.engine.functional.Functional
        the convolutional base, on top of which densely connected layers will
        be connected
    last_output : keras.engine.keras_tensor.KerasTensor
        output of last_layer
    compiled : boolean
        if false, the model will not get compiled. default is true.
    learning_rate : float
        the learning rate to be used, when compiled is true. default is +1e-4.
    returns
    -------
    model : keras.engine.functional.Functional
        final compiled model
    '''
    
    inputs = pretrained_model.input
    
    x = Flatten()(last_output)
    x = Dense(units = 1024, activation = "relu")(x)
    x = Dropout(.2)(x)
    
    outputs = Dense(units = 1, activation = "sigmoid")(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    if compiled:
        model.compile(optimizer = RMSprop(learning_rate = learning_rate),
                      loss = "binary_crossentropy",
                      metrics = ["accuracy"])
    
    return model

uncompiled_model = create_model(
    pretrained_model = pretrained_model,
    last_output = last_output,
    compiled = False)

best_learning_rate, history_learning_rate = adjust_learning_rate(
    uncompiled_model = uncompiled_model,
    training_generator = training_generator)

model = create_model(
    pretrained_model = pretrained_model,
    last_output = last_output,
    compiled = True,
    learning_rate = best_learning_rate)

history_transfer_learning = model.fit(train_generator,
                                      validation_data = validation_generator,
                                      steps_per_epoch = 100,
                                      epochs = 20,
                                      validation_steps = 50,
                                      verbose = 2,
                                      callbacks = [myCallback()])