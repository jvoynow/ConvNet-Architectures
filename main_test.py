# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:08:27 2020

@author: voyno
"""

from Data import load_data
from Residual_network import ResNet

import pandas as pd
import numpy as np

data = load_data()
train, test = data.get_data()

data = load_data()
train, test = data.get_data()

num_layers=32
resnet = ResNet(num_layers=num_layers)
train_history = resnet.train(train, test, augment=True)

train_history_data = np.array(list(train_history.history.values()))
train_history_keys = list(train_history.history.keys())

df = pd.DataFrame(data=train_history_data.T, columns=train_history_keys)
df.to_csv('Training_data/resnet_' + str(num_layers) + ".csv", index=None)