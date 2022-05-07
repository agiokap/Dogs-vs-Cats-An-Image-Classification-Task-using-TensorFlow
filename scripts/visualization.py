# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:23:53 2022

@author: agiokap
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_history(path):
    with open(path) as history_json:
        history = json.load(history_json)
    return history

def acc_val_plots(history, from_json = False, augmentation = False, transfer_learning = False):
    
    if from_json:
        history = load_history(history)
        
    acc, val_acc = history["accuracy"], history["validation_accuracy"]
    loss, val_loss = history["loss"], history["validation_loss"]
    
    num_epochs = range(len(acc))
    
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1)
    fig.suptitle("Accuracy and Loss ({} Augmentation/{} Transfer Learning)".format(("with" if augmentation == True else "without"),
                                                                                   ("with" if transfer_learning == True else "without")))
    
    ax_acc.plot(num_epochs, acc)
    ax_acc.plot(num_epochs, val_acc)
    plt.legend(["Training", "Validation"])
    ax_acc.set_ylabel("Accuracy")
    
    ax_loss.plot(num_epochs, loss)
    ax_loss.plot(num_epochs, val_loss)
    plt.legend(["Training", "Validation"])
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlabel("Epoch")

def best_lr(history, from_json = False):
    if from_json: history = load_history(history)
    loss, lr = history["loss"], history["lr"]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    ax2.set_xticks(tuple([epoch for epoch in range(0, 110, 10)]))
    ax1.plot(lr, loss)
    ax1.legend("Learning Rate")
    plt.title("Loss per Learning Rate per Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Learning Rate")
    ax2.set_xlabel("Epoch")
