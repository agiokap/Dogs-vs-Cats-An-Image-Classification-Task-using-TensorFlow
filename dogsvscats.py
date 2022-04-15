# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:54:26 2022

@author: agiokap
"""

# unzip the files, containing the data, into specific directories
def unzip(zip_location, extract_to):
    zipfile.ZipFile(zip_location, "r").extractall(extract_to)

def mkdrs(parent, children, source, split, unzip = False):
    """
    Parameters
    ----------
    parent: string
        The full path to the directory, where the training and testing subdirectories will be created.
    children: tuple (string)
        A number of classes-dimensional string tuple of classes' names, to be used when creating child
        subdirectories into parent directory. Note that coherence between child and source has to be retained.
        E.g., if child[i] regards class_i, then source[i] has also to be the full path to the directory, where
        the data, regarding class_i, is stored.
    source: tuple (string)
        A number of classes-dimensional string tuple, containing full paths, each of which directs to
        the directories, where the data regarding a specific class is stored. Note that two paths have to
        be included, if there are two target classes, i.e. when the task at hand is binary image classification,
        and three or more, if the task at hand is a multi-class classification.
    split: float
        A float number, ranging from 0. to 1., indicating the splitting rule. E.g., if split = .7, then 70%
        of the data, contained into each of the directories given by source, will be allocated to the training
        data, and the rest 30% to the testing data.
    unzip: string (default = False)
        A full path to the directory, where the zipped file that contains the data is located. Note that, if used,
        it has not to contradict the rest of the arguments. By default, the extraction path is chosen to be equal to
        the "parent" argument (see above), with no custom option.
    Returns
    -------
    None
    """
    
    # not False (any; a full path to the zipped file, containing the dataset)
    if unzip: unzip(zip_location = unzip, extract_to = parent)
    
    # initialize list to store valid filenames for each class[i] = children[i]
    filenames = []
    
    for i in range(len(children)):
        # create subdirectories of training and testing data for class[i] = children[i]
        os.makedirs(os.path.join(parent, "Training", children[i]))
        os.makedirs(os.path.join(parent, "Testing", children[i]))
        
        # collect valid filenames from source[i]
        for f in os.listdir(source[i]):
            if os.path.getsize(os.path.join(source[i], f)): # if file size != 0
                filenames.append(f)
            else: # if file size == 0
                print(f"{f} has zero length. ignoring...")
                
        # shuffle filenames
        filenames_shuffled = random.sample(filenames, len(filenames))
        
        # split shuffled filenames into training and testing datasets, using the split argument as the splitting rule
        split_point = int(split * len(filenames))
        filenames_train = filenames_shuffled[:split_point]
        filenames_test = filenames_shuffled[split_point:]
        
        # copy valid train and test files from source[i] to children[i]
        for f_train in filenames_train:
            copy_from = os.path.join(source[i], f_train)
            paste_to = os.path.join(parent, "training", children[i], f_train)
            copyfile(copy_from, paste_to)
            
        for f_test in filenames_test:
            copy_from = os.path.join(source[i], f_test)
            paste_to = os.path.join(parent, "testing", children[i], f_test)
            copyfile(copy_from, paste_to)

def data_generators(training_path, validation_path,
                    # flow_from_directory arguments
                    batch_size, class_mode, target_size,
                    # ImageDataGenerator parameters
                    rescale = None,
                    # augmentation (default: without augmentation)
                    rotation_range = 0, width_shift_range = 0.0, height_shift_range = 0.0,
                    shear_range = 0.0, zoom_range = 0.0,
                    horizontal_flip = False, fill_mode = "nearest"):
    
    training_generator = ImageDataGenerator(rescale = rescale,
                                            rotation_range = rotation_range,
                                            width_shift_range = width_shift_range,
                                            height_shift_range = height_shift_range,
                                            shear_range = shear_range,
                                            zoom_range = zoom_range,
                                            horizontal_flip = horizontal_flip,
                                            fill_mode = 'nearest').flow_from_directory(directory = training_path,
                                                                                       batch_size = batch_size,
                                                                                       class_mode = class_mode,
                                                                                       target_size = target_size)
                                                                                       
    validation_generator = ImageDataGenerator(rescale = rescale).flow_from_directory(directory = validation_path,
                                                                                     batch_size = batch_size,
                                                                                     class_mode = class_mode,
                                                                                     target_size = target_size)
    
    return training_generator, validation_generator

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

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(learning_rate = +1e-3),
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

history = model.fit(training_generator,
                    steps_per_epoch = 100,
                    epochs = 100,
                    validation_data = validation_generator,
                    validation_steps = 50,
                    verbose = 2)

import matplotlib.pyplot as plt

acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
loss, val_loss = history.history["loss"], history.history["val_loss"]

num_epochs = range(len(acc))

plt.plot(num_epochs, acc)
plt.plot(num_epochs, val_acc)
plt.title("Training and validation Accuracy")
plt.legend(["Training", "Validation"])
plt.figure()

plt.plot(num_epochs, loss)
plt.plot(num_epochs, val_loss)
plt.title("Training and validation Loss")
plt.legend(["Training", "Validation"])
plt.figure()
