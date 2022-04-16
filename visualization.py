# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:23:53 2022

@author: agiokap
"""

import json
import matplotlib.pyplot as plt

path = r"C:\Users\agiop\Documents\Data Science\Deep Learning with Python\TensorFlow Developing\dogsvscats\history.json"
with open(path) as history_json:
    history = json.load(history_json)

acc, val_acc = history["accuracy"], history["val_accuracy"]
loss, val_loss = history["loss"], history["val_loss"]

num_epochs = range(len(acc))

plt.plot(num_epochs, acc)
plt.plot(num_epochs, val_acc)
plt.title("Training and validation accuracy")
plt.legend(["Training", "Validation"])
plt.figure()

plt.plot(num_epochs, loss)
plt.plot(num_epochs, val_loss)
plt.title("Training and validation loss")
plt.legend(["Training", "Validation"])
plt.figure()