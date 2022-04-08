# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:54:26 2022

@author: agiokap
"""

import sys
import zipfile
import os
import numpy as np
import random
from shutil import copyfile

def in_virtual_environment():
    base, venv = sys.base_prefix, sys.prefix
    flag = venv == base
    if not flag:
        return print(not flag, venv)
    else:
        return print(flag, base)
    
in_virtual_environment() # this should be True

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
        A {number of classes}-dimensional string tuple of classes' names, to be used when creating child
        subdirectories into parent directory. Note that coherence between child and source has to be retained.
        E.g., if child[i] regards class_i, then source[i] has also to be the full path to the directory, where
        the data, regarding class_i, is stored.
    source: tuple (string)
        A {number of classes}-dimensional string tuple, containing full paths, each of which directs to
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
    unzip(zip_location = unzip, extract_to = parent) if unzip else None
    
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
            
mkdrs(parent = "C:/[...]/Documents/Data Science/Deep Learning with Python/TensorFlow Developing/dogsvscats/PetImages",
      children = ("Dogs", "Cats"),
      source = ("C:/[...]/Documents/Data Science/Deep Learning with Python/TensorFlow Developing/dogsvscats/PetImages/Dog",
                "C:/[...]/Documents/Data Science/Deep Learning with Python/TensorFlow Developing/dogsvscats/PetImages/Cat"),
      split = .9)
