### Generate Model


import tensorflow as tf
import subprocess
import re
import numpy as np
import mltk

import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

from tensorflow.keras.backend import clear_session
import gc





PROFILE_ERROR_FLAG = 0

def get_ondevice_hardware_attributes(m,ds):
        
        def representative_dataset():
            for input_value in tf.data.Dataset.from_tensor_slices(ds[0]).batch(1).take(100):
                yield [tf.dtypes.cast(input_value, tf.float32)]

       

        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #converter.inference_input_type = tf.uint8
        #converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()

        with open("test.tflite", 'wb') as f:
            f.write(tflite_quant_model)

        command = "mltk profile test.tflite"
        log = ""
        # Execute the command in the system shell and capture output
        try:
            completed_process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            log = completed_process.stdout
            print(log)

        except subprocess.CalledProcessError as e:
            print("Error:", e)

        return log



def parse_ondevice_hardware_attributes(log):

    global PROFILE_ERROR_FLAG

    ram_value = re.search(r'RAM, Runtime Memory Size \(bytes\): (\d+\.\d+)(k|M)', log)
    flash_value = re.search(r'Flash, Model File Size \(bytes\): (\d+\.\d+)(k|M)', log)
    mac_value = re.search(r'Multiply-Accumulate Count: (\d+\.\d+)(k|M)', log)

    
    if PROFILE_ERROR_FLAG == 0 :
        ram = -1
        flash = -1
        macc = -1

    else :

        ram =float("inf")
        flash = float("inf")
        macc = float("inf")

    if ram_value:
        ram_size, ram_unit = ram_value.group(1), ram_value.group(2)
        ram = float(ram_size)
        if ram_unit == 'M':
            ram *= 1000000
        elif ram_unit == 'k':
            ram *= 1000

    if flash_value:
        flash_size, flash_unit = flash_value.group(1), flash_value.group(2)
        flash = float(flash_size)
        if flash_unit == 'M':
            flash *= 1000000
        elif flash_unit == 'k':
            flash *= 1000

    if mac_value:
        mac_size, mac_unit = mac_value.group(1), mac_value.group(2)
        macc = float(mac_size)
        if mac_unit == 'M':
            macc *= 1000000
        elif mac_unit == 'k':
            macc *= 1000


    if (ram != -1) or (flash != -1) :
        PROFILE_ERROR_FLAG = 1
        
        
    return ram, flash, macc 

def evaluate_hardware_requirements(model, ds):
    log = get_ondevice_hardware_attributes(model,ds)
    return parse_ondevice_hardware_attributes(log)



class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_f1_score', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name='true_positives', shape=(num_classes,), initializer='zeros'
        )
        self.false_positives = self.add_weight(
            name='false_positives', shape=(num_classes,), initializer='zeros'
        )
        self.false_negatives = self.add_weight(
            name='false_negatives', shape=(num_classes,), initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # One-hot encode y_true if it's not already
        if y_true.shape[-1] != self.num_classes:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)

        # Convert predictions to binary using argmax
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=self.num_classes)

        # Compute true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Compute precision and recall
        precision = self.true_positives / (
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )

        # Compute F1 per class
        f1_per_class = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        # Return macro F1-score
        return tf.reduce_mean(f1_per_class)

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))






def BuildModelwithSpecs(k,c,num_class = 2 , ds = None ,  input_shape = (1,60,6),learning_rate = 0.0001 , lossf=1):


    print("@          k,c,num_class : ",k,c,num_class)

    kernel_size = 3
    pool_size = 2
    

    non_strideable = False


    inputs = tf.keras.Input(input_shape)

    
    # convolutional base
    n = k
    multiplier = 1.5

    # first convolutional layer
    #x = tf.keras.layers.Conv1D(n, kernel_size, activation='relu', padding='same')(inputs)

    x = tf.keras.layers.SeparableConv1D(n, kernel_size, activation='relu', padding='same')(inputs)

   
    

    # adding cells
    for i in range(1, c + 1) :
        if x.shape[1] <= 1 or x.shape[2] <= 1 :
            non_strideable = True
            break

        n = int(n * multiplier)
        
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
        x =  tf.keras.layers.SeparableConv1D(n, kernel_size, activation='relu', padding='same')(x)
        
    # classifier
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    n = int(n * multiplier)
    x = tf.keras.layers.Dense(n, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)

    # model building
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'sparse_categorical_crossentropy'
    if lossf == 1 : # category
        loss = 'categorical_crossentropy'

    


  

    model.compile(optimizer=opt,
            loss=loss ,
            metrics=['accuracy', tf.keras.metrics.Precision(),
                              tf.keras.metrics.Recall(),
                              tfa.metrics.F1Score(num_classes=num_class,
                                                  average='macro',
                                                  threshold=0.5),
                              MacroF1Score(num_classes=num_class),
                            ])

    model.summary()

    ram, flash, macc  = evaluate_hardware_requirements(model,ds)

    return model, ram, flash, macc 



#def ModelTraning(model,train_data, val_data, epochs=3):
def ModelTraning(model,train_ds,val_ds = None , epochs = 3):

    y_label = train_ds[1]

    class_indices = np.argmax(y_label, axis=1)
    classes = np.unique(class_indices)

    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=class_indices
    )

    class_weight_dict = dict(enumerate(class_weights))


    print(class_weight_dict)
    
    hist = []
    file_path = "best_model.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_f1_metric', verbose=1, save_best_only=True, mode='max')
    #early =  tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=2)
    redonplat =  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_f1_metric", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint,  redonplat]  # early
    #class_weight=class_weight_dict,

    if val_ds == None :
        
        #hist = model.fit(train_ds[0],train_ds[1] ,epochs=epochs,  verbose =True, validation_split=0.15,callbacks=callbacks_list) 
        hist = model.fit(train_ds[0],train_ds[1] ,epochs=epochs,  verbose =True, validation_split=0.15 , ) 
    else :
        hist = model.fit(train_ds[0],train_ds[1] ,epochs=epochs,  verbose =True, validation_data=(val_ds[0], val_ds[1]),     callbacks=[]) 
    max_val_acc = np.around(np.amax(hist.history['val_macro_f1_score']), decimals=3)

    del model 
    clear_session()
    gc.collect()

    return max_val_acc



def CheckFeasible(constraints_specs,current_specs):
    ram = current_specs["ram"] <= constraints_specs["ram"]
    flash = current_specs["flash"] <= constraints_specs["flash"] 
    macc = current_specs["macc"] <= constraints_specs["macc"]
    return ram and flash and macc





