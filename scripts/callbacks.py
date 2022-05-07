# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:14:50 2022

@author: agiokap
"""

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

def adjust_learning_rate(uncompiled_model, training_generator):
    '''
    Parameters
    ----------
    uncompiled_model : keras.engine.functional.Functional
        the uncompiled model, whose learning rate is to be tuned.
    training_set: numpy.ndarray
        the training data
    Returns
    -------
    history : numpy.ndarray
        the history log, containing information about the learning rate and the
        corresponding loss. used to determine the best learning rate value.
    '''
    
    learning_rate_schedule = LearningRateScheduler(
        lambda epoch: +1e-8 * (10 ** (np.arrange(epoch / 20))))
    
    optimizer = SGD(momentum = .9)
    
    uncompiled_model.compile(optimizer = optimizer, loss = "binary_crossentropy")
    
    history_learning_rate = uncompiled_model.fit(training_generator,
                                                 epochs = 100,
                                                 callbacks = [learning_rate_schedule])
    
    history_df = pd.DataFrame(history)
    min_loss_index = history_df.loc[:, "loss"].argmin()
    best_learning_rate = history_df.loc[min_loss_index, "lr"]
    
    return best_learning_rate, history_learning_rate

class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs = {}): 
        if(logs.get('acc') > .95):   
            print("\naccuracy > 95% => stop training")
        self.model.stop_training = True