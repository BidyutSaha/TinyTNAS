

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
train_data = np.load('all_datasets/train_data_mitbih.npz')
X = train_data['features']
Y = train_data['labels']

# Load test data
test_data = np.load('all_datasets/test_data_mitbih.npz')
X_test = test_data['features']
Y_test = test_data['labels']




# Time bound Hardware Aware Neural Architecture Serach ################################################## 

# Creating a instance of TinyTNAS and Neural Architecture Searching ---

# -  `train_ds` : training dataset as (X,Y)
# -  `val_ds`   : validation dataset as (X,Y)
# - `input_shape` : input shape of the features
# - `num_class`   : number of classes
# - `learning_rate` : learning rate for the optimization
# - `constraints_specs` : hardware constarins for  ram, flash and mac


batch_size=32
input_shape = (X.shape[1:])
learning_rate = 0.001

train_ds = (X,Y)
val_ds = (X_test,Y_test)

num_class = 5
# Set `lossf` based on the format of your target labels:
# - If your target labels are one-hot encoded, use `lossf = 0` for categorical crossentropy.
# - If your target labels are integers (i.e., class indices), use `lossf = 1` for sparse categorical crossentropy.
lossf = 0


constraints_specs= {"ram"   : 1024*20,    # 'ram' in bytes
                    "flash" : 1024*64,    # 'flash' in bytes
                    "macc"  : 60*1000 }


algo = TinyTNAS(train_ds=train_ds,val_ds=None,input_shape = input_shape,
                num_class = num_class,learning_rate = learning_rate, constraints_specs= constraints_specs)

start_time_sec = datetime.now()

# Perform Neural Architecture Search ---
# The `algo.search` method explores various neural network architectures with the following parameters and suggest the best among them:
# - `epochs`: Number of epochs to train each candidate architecture. More epochs generally lead to better-trained models.
# - `search_time_minute`: The maximum duration (in minutes) allowed for the search process. The algorithm will attempt to find the best architecture within this time frame.


results = algo.search(epochs=5,  lossf = lossf , search_time_minute=2)
end_time_sec = datetime.now()
elapsed_time_minute = (end_time_sec - start_time_sec)/60.0
print(f"Elapsed time minute: {elapsed_time_minute}")



best_k = results[0]
best_c = results[1]
best_acc = results[2]
best_ram = results[3]
best_flash = results[4]
best_macc = results[5]
architecure_explored_count = algo.explored_model_count
architecture_search_path = algo.feasible_solutions
architecure_explored = algo.explored_model_configs
infeasible_architectures = algo.infeasible_configarations



data  = {
    "best_k" : best_k,
    "best_c" : best_c,
    "best_acc" : best_acc,
    "best_ram" : best_ram,
    "best_flash" : best_flash,
    "best_macc" : best_macc,
    "architecure_explored_count" : architecure_explored_count,
    "architecture_search_path" : architecture_search_path,
    "architecure_explored" : architecure_explored,
    "infeasible_architectures" : infeasible_architectures

}

print(data)





#Full Training with the Best Architecture #########################################################

# After executing the algorithm, the best model identified will be trained with sufficient epochs
# to maximize its generalization capabilities.

file_path = "bestmodel.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early


best_model,_,_,_ = BuildModelwithSpecs(k=best_k,c=best_c,num_class = num_class , ds = train_ds ,  input_shape = input_shape,learning_rate = learning_rate , lossf=lossf)
print(best_model)
max_val_Acc = ModelTraning(best_model,train_ds,val_ds , epochs = 500, )
print(max_val_Acc)







