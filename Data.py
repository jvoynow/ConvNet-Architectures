# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 14:38:25 2020

@author: voyno
"""

from keras.datasets import cifar10
from keras.utils import to_categorical

import numpy as np


class load_data:
    
    def __init__(self, normalize=True):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
        if normalize:
            x_train = self.normalize_(x_train)
            x_test = self.normalize_(x_test)
        
        self.data = [x_train, y_train], [x_test, y_test]
    
    
    @staticmethod
    def normalize_(x):
        numerator = x - np.min(x)
        denominator = np.max(x) - np.min(x)
        return numerator/ denominator
        
    
    def get_data(self):
        return self.data