

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd

from TinyTNAS import TinyTNAS
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from ModelBank import *




# DATASET ##############################################################################################

# Load train data
train_data = np.load('./all_datasets/train_data_ecg.npz')
X = train_data['features']
Y = train_data['labels']

# Load test data
test_data = np.load('./all_datasets/test_data_ecg.npz')
X_test = train_data['features']
Y_test  = train_data['labels']


print(X.shape, Y.shape , X_test.shape, Y_test.shape)





batch_size=32
input_shape = (X.shape[1:])
learning_rate = 0.001

train_ds = (X,Y)
val_ds = (X_test,Y_test)

num_class = 4
# Set `lossf` based on the format of your target labels:
# - If your target labels are one-hot encoded, use `lossf = 1` for categorical crossentropy.
# - If your target labels are integers (i.e., class indices), use `lossf = 0` for sparse categorical crossentropy.
lossf = 1




#Full Training with the Best Architecture #########################################################

# After executing the algorithm, the best model identified will be trained with sufficient epochs
# to maximize its generalization capabilities.

best_k = 32
best_c  =  1
num_class 

file_path = "bestmodel.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=10, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early


best_model,_,_,_ = BuildModelwithSpecs(k=best_k,c=best_c,num_class = num_class , ds = train_ds ,  input_shape = input_shape,learning_rate = learning_rate , lossf=lossf)
max_val_Acc = ModelTraning(best_model,train_ds,val_ds , epochs = 500, )
print(max_val_Acc)






