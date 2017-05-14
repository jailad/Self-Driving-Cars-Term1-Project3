
# coding: utf-8

# # Overview
# * A lot of the experimentation for this project was performed within 'Term1Project3BehaviorCloning' and 'SideExperiments''
# * However, as per Rubric requirements, there is a need for a finalized 'model' file, which contains the implementation.
# * Therefore, this file was created essentially as a condensed version of the above notebooks

# In[3]:


# Common Imports

import resource
import numpy as np
import csv
import time # Used for calculating the run-time of code segments
import matplotlib.pyplot as plt # Useful for generating training plots
# Useful for inline plot generation
get_ipython().magic('matplotlib inline')
import sklearn
from sklearn.model_selection import train_test_split
import datetime

# Initial Setup for Keras
from keras.layers.convolutional import Convolution2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten, Dropout, Lambda, Input, Cropping2D, ELU
from keras import backend as K
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

import pandas # For generating descriptive statistics
from pandas.tools.plotting import scatter_matrix
import cv2 # Include Open CV


# In[4]:


# Shared Constants

const_separator_line = "--------------------------------"
const_csv_filename = 'driving_log.csv'
const_image_foldername = 'IMG/'
const_udacity_data_folder_name = 'data/fromudacity/'
const_my_data_fast_track1_clockwise_folder_name = 'data/fast_track1clockwise/'
const_my_data_fast_track1_counterclockwise_folder_name = 'data/fast_track1counterclockwise/'
const_my_data_track1_curves_cw_folder_name = 'data/mydata_track1_curves_cw/'
const_my_data_track1_curves_ccw_folder_name = 'data/mydata_track1_curves_ccw/'


const_validation_data_ratio = 0.5 # Split off 20% of samples for validation

# Named constants for important parameter values

const_dropout_probability = 0.2
const_num_epochs = 3
const_activation_function = 'relu'
const_padding_strategy = 'valid'
const_padding_strategy_same ='same'
const_loss_function = 'mse'
const_optimizer_function = 'adam'
const_model_filename_prefix = 'model_'
const_model_filename_postfix = '.h5'


# In[5]:



# Useful to selectively turn on / off logging at different levels

const_info_log_enabled = False
def infoLog(logMessage, param_separator=None):
    if const_info_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)

const_debug_log_enabled = True
def debugLog(logMessage, param_separator=None):
    if const_debug_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)
        
const_warning_log_enabled = True
def warningLog(logMessage, param_separator=None):
    if const_warning_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)
        
const_error_log_enabled = True
def errorLog(logMessage, param_separator=None):
    if const_error_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)



# In[6]:


# Convenience method to get and Print current memory usage

def print_memory_usage():
    infoLog("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
print_memory_usage()


# In[7]:


# Convenience method to get Current Time

def get_current_time():
    current_time = time.time()
    return current_time

debugLog(get_current_time())


# In[8]:


# Return Current time in human readable format

def get_current_human_readable_time():
    return datetime.datetime.now().isoformat()

debugLog(get_current_human_readable_time())


# In[9]:


# Convenience method to get Print Time Difference

def print_time_diff(start_time, end_time, activity_label="task"):
    time_difference = end_time - start_time
    debugLog("Execution time for " + activity_label + " : " + str(time_difference) + " seconds")
    return time_difference


# In[10]:


# Convenience functions to generate different types of Keras model architectures

# Defining convenience parameters for different network architectures
# C = Convolution
# A = Activation
# P = Pooling
# FL = Flatten
# FC = Fully Connected
# D = Dropout

# Base Model with Preprocessing
def get_base_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    #     model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    return model

# From - https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# From - http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# Changes relative to above architecture, if applicable, have been called out at the individual layers

def get_nvidia_model():
    model_desc_name = 'nvidia_cnn_model'
    # Normalization/ Preprocessing
    model = get_base_model()
    
    #CA24_5_2_VALID
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    
    #CA36_5_2_VALID
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    
    #CA48_5_2_VALID
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    
    #CA64_3_1_VALID
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    
    #CA64_3_1_VALID
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    
    #FL
    model.add(Flatten())
    
    #FCA1164
    model.add(Dense(1164))
    model.add(Activation(const_activation_function))
    
    #FCA100
    model.add(Dense(100))
    model.add(Activation(const_activation_function))
    
    #FCA50
    model.add(Dense(50))
    model.add(Activation(const_activation_function))
    
    #D-0.7
    # Dropout was not shown in the original architecture, but I added it to prevent over fitting
    model.add(Dropout(const_dropout_probability))
    
    #FC1
    model.add(Dense(1))
    
    model.compile(loss=const_loss_function, optimizer=const_optimizer_function, metrics=['accuracy','precision','recall'])

    return model, model_desc_name





# In[11]:


# Convenience method which takes a Keras Model, and prints it's architecture

def print_model_architecture(keras_model, model_label):
    debugLog("Model Architecture for " + model_label)
    layers = keras_model.layers
    for index, layer in enumerate(layers):
        debugLog("Layer " + str(index))
        debugLog(layer)
        
model, model_name = get_nvidia_model()
print_model_architecture(model, model_name)


# In[12]:


# Convenience method which takes a Keras History object and prints it

def plot_keras_history_object(keras_history_object, model_label):
    plt.plot(keras_history_object.history['loss'])
    plt.plot(keras_history_object.history['val_loss'])
    plt.title('Model : ' + model_label +  ' -> mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# In[13]:



# Convenience method to convert a long absolute path to relative path

sample_absolute_path = "/Users/jai/Desktop/BitBucket/UDSDC/GitHubSubmissions/Self-Driving-Cars-Term1-Project3-BehavioralCloning/data/fast_track1clockwise/IMG/center_2017_05_05_13_51_02_269.jpg"

def get_relative_path(param_absolute_path):
    string_split = param_absolute_path.split('/')
    string_split_last4 = string_split[-4:]
    relative_path = "/".join(string_split_last4)
    return relative_path

debugLog(get_relative_path(sample_absolute_path))


# # Using a Generator approach for loading, processing and training data from CSV file

# In[14]:


# Convenience method to read a CSV file and return left, right, center image paths, and parameters like steering angle, throttle etc.

def load_csv_file_all_data(filename, param_offset = 0.01):

    debugLog("Started Data Reading :" + filename)
    paths = list()
    steering_angles = list()
    with open(filename, 'r') as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.read(1024))
        infoLog("Has Header is " + str(has_header) + " for :" + filename)
        csvfile.seek(0)  # rewind
        header = csv.reader(csvfile)
        if has_header:
            next(header)  # skip header row
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for data_row in reader:
                center = data_row[0].strip() # Stripping is important in case there are any unintended leading / trailing spaces in the data
                left = data_row[1].strip()
                right = data_row[2].strip()
                steering = data_row[3]
                throttle = data_row[4]
                brake = data_row[5]
                speed = data_row[6]
                                
                # Filtering Logic
                steering_angle = float(steering)
                
                steering_left = steering_angle + param_offset
                steering_right = steering_angle - param_offset

                # Include all images which are under represented
                if (steering_angle < -0.1 and steering_angle > float('-inf') ) or (steering_angle > 0.1 and steering_angle < float('inf')):
                    paths.append(left)
                    paths.append(center)
                    paths.append(right)
                    
                    steering_angles.append(steering_left)
                    steering_angles.append(steering)
                    steering_angles.append(steering_right)

                # Include 50% of images which are over represented
                else:
                    index = np.random.randint(0, 100)
                    if index >= 0 and index < 33:
                        paths.append(left)
                        paths.append(center)
                        paths.append(right)
                    
                        steering_angles.append(steering_left)
                        steering_angles.append(steering)
                        steering_angles.append(steering_right)
                        
    debugLog("Ended Data Reading :" + filename)
    return paths, steering_angles


print_memory_usage()
paths, steering_angles = load_csv_file_all_data('data/fromudacity/driving_log.csv')
print_memory_usage()

# Separate out data into training and validation
X_train, X_validation, y_train, y_validation = train_test_split(paths, steering_angles, test_size=const_validation_data_ratio, random_state=5)

debugLog(" # Training images : " + str(len(X_train)))
debugLog(" # Validation images : " + str(len(X_validation)))
debugLog(" # Training labels : " + str(len(y_train)))
debugLog(" # Validation labels : " + str(len(y_validation)))


# In[16]:


# Writing a generator which yields an iterable 'batch_size' number of images from the above samples

def load_images_generator(filepaths, labels, batch_size, prefix=None):
    # The reason for infinity iteration -> http://stackoverflow.com/questions/37798410/why-does-this-python-generator-have-no-output-according-to-keras
    while True:
        offset = 0
        for offset in range(0,len(filepaths),batch_size):
            batch_images_paths = filepaths[offset:offset+batch_size]
            batch_images = list()
            batch_labels = labels[offset:offset+batch_size]
            for image_path in batch_images_paths:
                if prefix:
                    image_path = prefix + image_path
                else:
                    image_path = get_relative_path(image_path)
                image = cv2.imread(image_path)
                batch_images.append(image)
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield sklearn.utils.shuffle(batch_images, batch_labels)


# In[17]:


# Convenience method to train a model given a specified CSV file path to the data set
# The model can be a fresh ( randomly initialized ) model, or a model which has been saved and restored from the Disk
# After the completion of training, the best model is saved to disk and also returned
# the suffix alldata represents the fact that we are using Left, Right and Center Data to do the training and validation

def train_model_for_csv_file_alldata(param_model, param_model_desc_name, csv_file_path, param_epochs = 2, param_steering_offset=0.02, param_batch_size = 64 ):
        
    debugLog(csv_file_path)
    
    path_prefix = 'data/fromudacity/'
    # TODO: A bit awkward to separate Udacity Data From Data Generated by me, but works for now
    if csv_file_path != const_udacity_data_csv_path:
        path_prefix = None
        debugLog("path_prefix is None because this is a non-udacity data file.")
    else:
        debugLog("path_prefix is : " + csv_file_path)
    
    # Load and Filter the Data - Only keep the Data we care about ( randomly drop oversampled data ) and return filtered arrays
    paths, steering_angles = load_csv_file_all_data(csv_file_path, param_steering_offset)    
    
    
    # Separate out data into training and validation
    X_train, X_validation, y_train, y_validation = train_test_split(paths, steering_angles, test_size=const_validation_data_ratio, random_state=5)
    debugLog(" # Training images : " + str(len(X_train)))
    debugLog(" # Validation images : " + str(len(X_validation)))
    debugLog(" # Training labels : " + str(len(y_train)))
    debugLog(" # Validation labels : " + str(len(y_validation)))
    
    # Start training process

    # Define Checkpoint and Early Stopping
    model_filepath = const_model_filename_prefix + model_desc_name + str(get_current_human_readable_time()) + "-{epoch:02d}-{val_acc:.2f}" + const_model_filename_postfix
    model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0)
    model_callbacks_list = [model_checkpoint, early_stopping]

    train_generator = load_images_generator(X_train, y_train, param_batch_size, path_prefix)
    validation_generator = load_images_generator(X_validation, y_validation, param_batch_size, path_prefix)
    start_time = get_current_time()
    debugLog(get_current_human_readable_time())
    model_history = param_model.fit_generator(train_generator, samples_per_epoch = len(X_train), validation_data = validation_generator, nb_val_samples = len(X_validation), nb_epoch = param_epochs, callbacks = model_callbacks_list)
    debugLog(model_history.history.keys())
    end_time = get_current_time()
    debugLog(get_current_human_readable_time())
    print_time_diff(start_time,end_time,"Training " + param_model_desc_name + " for " + str(param_epochs) + " epochs")    
    # Generate plot of accuracy over epochs
    plot_keras_history_object(model_history,param_model_desc_name)
    
    val_loss_str = "val_loss_" + str(model_history.history.get("val_loss", "unknown"))
    model_save_filename = const_model_filename_prefix + model_desc_name + str(get_current_human_readable_time()) + val_loss_str + const_model_filename_postfix
    param_model.save(model_save_filename)
    debugLog("Saved model with name : " + model_save_filename)

    # Return the Model
    return param_model
        


# In[ ]:

# Train the 'Nvidia Model'

# Create a new model from scratch
# nvidia_model, model_desc_name = get_nvidia_model()

# or, Restore one from Disk
model_path = 'model.h5'
model_desc_name = 'nvidia_cnn_model'
nvidia_model = load_model(model_path)

# Print out the architecture
debugLog(model_path)
print_model_architecture(nvidia_model, model_desc_name)

# Train Repeatedly for different data sets

nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_udacity_data_csv_path, 5 , 0.2, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_fast_track1_clockwise_csv_path, 1 , 0.08, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_fast_track1_counter_clockwise_csv_path, 1 , 0.08, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_track1_curves_cw_csv_path, 1 , 0.08, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_track1_curves_ccw_csv_path, 1 , 0.08, 100)

