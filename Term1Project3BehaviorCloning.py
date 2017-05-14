
# coding: utf-8

# ## Project 3 - Behavioral Cloning

# * The main pipeline used to perform data analysis, training, and validation of the model for this project

# In[1]:


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


# In[2]:


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


# In[3]:


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


# In[4]:

# Convenience method to get and Print current memory usage

def print_memory_usage():
    infoLog("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
print_memory_usage()


# In[5]:

# Convenience method to get Current Time

def get_current_time():
    current_time = time.time()
    return current_time

debugLog(get_current_time())


# In[6]:

# Return Current time in human readable format

def get_current_human_readable_time():
    return datetime.datetime.now().isoformat()

debugLog(get_current_human_readable_time())


# In[7]:

# Convenience method to get Print Time Difference

def print_time_diff(start_time, end_time, activity_label="task"):
    time_difference = end_time - start_time
    debugLog("Execution time for " + activity_label + " : " + str(time_difference) + " seconds")
    return time_difference


# In[50]:

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

# Simple model for quick training and validation
def get_simple_model():
    model_desc_name = 'simple_model'
    model = get_base_model()
    model.add(Convolution2D(32, 3, 3, border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode=const_padding_strategy))
    model.add(Dropout(const_dropout_probability))

    model.add(Convolution2D(64, 3, 3, border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode=const_padding_strategy))
    model.add(Dropout(const_dropout_probability))

    model.add(Convolution2D(128, 3, 3, border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode=const_padding_strategy))
    model.add(Dropout(const_dropout_probability))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation(const_activation_function))

    model.add(Dense(1))
    
    model.compile(loss=const_loss_function, optimizer=const_optimizer_function, metrics=['accuracy','precision','recall'])

    return model, model_desc_name


# Returns an architecture which is based on the architecture which I used successfully for Project 2 - Behavioral Cloning
# Changes relative to above have been called out at the individual layers
def get_my_model():
    model_desc_name = 'my_model'
    model = get_base_model()
    
    #CAP5_16 - Pool Size increased to 4 from 2, to reduce number of parameters
    model.add(Convolution2D(16, 5, 5, border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode=const_padding_strategy))

    #CAP5_32 - Pool Size increased to 4 from 2, to reduce number of parameters
    model.add(Convolution2D(32, 5, 5, border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode=const_padding_strategy))
    
    #CAP2_256 - Depth reduced to 64 from 256, to reduce number of parameters
    model.add(Convolution2D(64, 5, 5, border_mode=const_padding_strategy))
    model.add(Activation(const_activation_function))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None, border_mode=const_padding_strategy))
    
    #FL
    model.add(Flatten())

    #FCA_240 -> 256
    model.add(Dense(256))
    model.add(Activation(const_activation_function))

    #FCA_168 -> 64
    model.add(Dense(64))
    model.add(Activation(const_activation_function))
        
    #D-0.7
    model.add(Dropout(const_dropout_probability))

    #FC_43 -> FC_1
    model.add(Dense(1))
    
    model.compile(loss=const_loss_function, optimizer=const_optimizer_function, metrics=['accuracy','precision','recall'])

    return model, model_desc_name


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

# From - https://github.com/commaai/research/blob/master/train_steering_model.py
def get_commaai_model():
    model_desc_name = 'commaai_cnn_model'
    model = get_base_model()

    model.add(Convolution2D(16, 8, 8, subsample=(1, 1), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(const_dropout_probability))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(const_dropout_probability))
    model.add(ELU())
    
    model.add(Dense(1))
    
    model.compile(loss=const_loss_function, optimizer=const_optimizer_function, metrics=['accuracy','precision','recall'])

    return model, model_desc_name

def get_inception_model():
    model_desc_name = 'inception_model'
    
    img_width = 160
    img_height = 320

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1)(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    
    infoLog(model_desc_name + " layers")
    
    for i, layer in enumerate(model.layers):
        debugLog("Layer : " + str(i) + " Layer Name:" + str(layer.name))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(loss=const_loss_function, optimizer=const_optimizer_function, metrics=['accuracy','precision','recall'])

    return model, model_desc_name


# In[54]:

# Convenience method which takes a Keras Model, and prints it's architecture

def print_model_architecture(keras_model, model_label):
    debugLog("Model Architecture for " + model_label)
    layers = keras_model.layers
    for index, layer in enumerate(layers):
        debugLog("Layer " + str(index))
        debugLog(layer)
        
model, model_name = get_simple_model()
print_model_architecture(model, model_name)


# In[8]:

# Convenience method which takes a Keras History object and prints it

def plot_keras_history_object(keras_history_object, model_label):
    plt.plot(keras_history_object.history['loss'])
    plt.plot(keras_history_object.history['val_loss'])
    plt.title('Model : ' + model_label +  ' -> mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# In[9]:

# Convenience method, which takes in array of filepaths, and labels, and plots a random images from the set

def visualize_image_from(filepaths, labels, file_path_prefix=None):
    index = np.random.randint(0, len(filepaths))   
    image_filepath = filepaths[index]
    if file_path_prefix:
        image_filepath = file_path_prefix + image_filepath
    else:
        image_filepath = get_relative_path(image_filepath)
    image_label = labels[index]
    image = cv2.imread(image_filepath)
    
    infoLog(image_filepath)
    infoLog(index)
    infoLog(image.shape) 
    infoLog(image.dtype) 
    infoLog(image_label)
    
    plt.title(str(image_label) + " || "+ image_filepath)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    


# In[10]:

# Convenience method which takes in a CSV file's path, and generates statistics for the same

def generate_stats_and_visualization(csv_file_name, has_header):
    dataset = None
    if has_header == False:
        dataset = pandas.read_csv(csv_file_name, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    else:
        dataset = pandas.read_csv(csv_file_name)

    debugLog("Start - Stats and Visualization for - FileName : " + csv_file_name, const_separator_line)

    debugLog("Shape = " + str(dataset.shape), const_separator_line)

    debugLog("Top 5 Rows = " + str(dataset.head(5)), const_separator_line)

    debugLog("Dataset Description \n" + str(dataset.describe()), const_separator_line)

    debugLog("Steering Angle Distributions \n",const_separator_line )
    debugLog(dataset.groupby('steering').size())

    debugLog("Box and Whisker Plots \n",const_separator_line)
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    debugLog("Histograms \n",const_separator_line)
    dataset.hist(bins=20)
    plt.show()
    debugLog("End - Stats and Visualization for - FileName : " + csv_file_name, const_separator_line)
    


# In[11]:


# Convenience method to convert a long absolute path to relative path

sample_absolute_path = "/Users/jai/Desktop/BitBucket/UDSDC/GitHubSubmissions/Self-Driving-Cars-Term1-Project3-BehavioralCloning/data/fast_track1clockwise/IMG/center_2017_05_05_13_51_02_269.jpg"

def get_relative_path(param_absolute_path):
    string_split = param_absolute_path.split('/')
    string_split_last4 = string_split[-4:]
    relative_path = "/".join(string_split_last4)
    return relative_path

debugLog(get_relative_path(sample_absolute_path))


# In[12]:

# Generate Statistics for - Udacity Data

const_udacity_data_csv_path = const_udacity_data_folder_name + const_csv_filename
const_udacity_data_images_folder_path = const_udacity_data_folder_name + const_image_foldername

generate_stats_and_visualization(const_udacity_data_csv_path, True)


# In[13]:

# Generate Statistics for - My Data - Fast Driving - Track 1 - Clockwise

const_my_data_fast_track1_clockwise_csv_path = const_my_data_fast_track1_clockwise_folder_name + const_csv_filename
const_my_data_fast_track1_clockwise_images_folder_path = const_my_data_fast_track1_clockwise_folder_name + const_image_foldername

generate_stats_and_visualization(const_my_data_fast_track1_clockwise_csv_path, False)


# In[14]:

# Generate Statistics for - My Data - Fast Driving - Track 1 - Counter Clockwise

const_my_data_fast_track1_counter_clockwise_csv_path = const_my_data_fast_track1_counterclockwise_folder_name + const_csv_filename
const_my_data_fast_track1_counter_clockwise_images_folder_path = const_my_data_fast_track1_counterclockwise_folder_name + const_image_foldername

generate_stats_and_visualization(const_my_data_fast_track1_counter_clockwise_csv_path, False)


# In[15]:

# Generate Statistics for - My Data - Fast Driving - Track 1 - Counter Clockwise

const_my_data_fast_track1_counter_clockwise_csv_path = const_my_data_fast_track1_counterclockwise_folder_name + const_csv_filename
const_my_data_fast_track1_counter_clockwise_images_folder_path = const_my_data_fast_track1_counterclockwise_folder_name + const_image_foldername

generate_stats_and_visualization(const_my_data_fast_track1_counter_clockwise_csv_path, False)


# In[16]:

# Generate Statistics for - My Data - Track 1 - Curves - Clockwise

const_my_data_track1_curves_cw_csv_path = const_my_data_track1_curves_cw_folder_name + const_csv_filename
const_my_data_track1_curves_cw_images_folder_path = const_my_data_track1_curves_cw_csv_path + const_image_foldername

generate_stats_and_visualization(const_my_data_track1_curves_cw_csv_path, False)


# In[17]:

# Generate Statistics for - My Data - Track 1 - Curves - Clockwise

const_my_data_track1_curves_ccw_csv_path = const_my_data_track1_curves_ccw_folder_name + const_csv_filename
const_my_data_track1_curves_ccw_images_folder_path = const_my_data_track1_curves_ccw_csv_path + const_image_foldername

generate_stats_and_visualization(const_my_data_track1_curves_ccw_csv_path, False)


# ## Summary of above dataset(s)

# * In general we can see from the above distributions, that for steering angle(s), we have a lot of data between -0.1 and + 0.1. Beyond a certain point, additional training will not really add much value to the network
# 
# * For these limits, or other dynamically defined limits, we can tune the data ingestion process to randomly drop samples for highly represented classes.

# ## Experiment 2: Using a Generator approach for loading, processing and training data from CSV file

# In[35]:


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


# In[37]:

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
    


# In[ ]:

visualize_image_from(X_train,y_train,'data/fromudacity/')


# In[ ]:

# Define training generator and validation generator

train_generator = load_images_generator(X_train,y_train,32,'data/fromudacity/')
validation_generator = load_images_generator(X_validation, y_validation,32, 'data/fromudacity/')


# In[45]:


# Convenience method to train a model given a specified CSV file path to the data set
# The model can be a fresh ( randomly initialized ) model, or a model which has been saved and restored from the Disk
# After the completion of training, the best model is saved to disk and also returned

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

# Train the 'Simple Model'

# Define the model
simple_model, model_desc_name = get_simple_model()

train_model_for_csv_file(simple_model, model_desc_name, const_udacity_data_csv_path)


# In[ ]:

# Train 'My Model'

my_model, model_desc_name = get_my_model()

train_model_for_csv_file(my_model, model_desc_name, const_udacity_data_csv_path)


# In[55]:

# Train the 'Nvidia Model'

# Create a new model from scratch
# nvidia_model, model_desc_name = get_nvidia_model()

# or, Restore one from Disk
model_path = 'model_nvidia_cnn_model2017-05-07T14:11:17.119973-00-0.12_success_refinement_pending.h5'
model_desc_name = 'nvidia_cnn_model'
nvidia_model = load_model(model_path)

# Print out the architecture
debugLog(model_path)
print_model_architecture(nvidia_model, model_desc_name)

# Train Repeatedly for different data sets

nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_udacity_data_csv_path, 1 , 0.2, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_fast_track1_clockwise_csv_path, 1 , 0.08, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_fast_track1_counter_clockwise_csv_path, 1 , 0.08, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_track1_curves_cw_csv_path, 1 , 0.08, 100)
# nvidia_model = train_model_for_csv_file_alldata(nvidia_model, model_desc_name, const_my_data_track1_curves_ccw_csv_path, 1 , 0.08, 100)


# In[ ]:

# Train the 'CommaAI Model'

# Define the model
commaai_model, model_desc_name = get_commaai_model()

train_model_for_csv_file(commaai_model, model_desc_name, const_udacity_data_csv_path)


# ## ( Success ) -  Experiment 3: Using Transfer Learning ( Inception V3 ) with Keras
# 
# Initial Failure = ValueError: Error when checking model target: expected dense_2 to have 4 dimensions, but got array with shape (32, 1)
# 
# Success in at least get the model running, and compiling was achieved by adding a Flattening layer to the end of the convolutional layers as obtained from Inception V3
# 
# Pending = Explore how to add preprocessing lambda layers to the Inception V3 model
# 

# In[ ]:


inception_model, model_desc_name = get_inception_model()

train_model_for_csv_file(inception_model, model_desc_name, const_udacity_data_csv_path)


# ## Experiment 4: Compare the above models
# 

# * Models to compare :
# 
# 
#     + Nvidia pipeline.
#     + CommaAI pipeline.
#     + My Project 2 pipeline.
#     + Simple pipeline.
# 

# * Comparison criteria :
# 
# 
#     + Training accuracy.
#     + Validation accuracy.
#     + Training loss.
#     + Validation loss.
#     + Training time.
#     + Any other parameters

# ## Experiment #: Review classes with worst performance, and best performance
# 

# ## Experiment #: Analysis of Data Distribution
# 

# * The results of the analysis / distribution below, show that ( as expected ), we have a lot of training data for the 'center' case(s) but very less data particularly for two scenarios :
# * 1. When steering angle value is negative.
# * 2. When steering angle value is at the nagative or positive extremities.
# * One way to improve performance, would be to take the already trained model ( 54% accuracy ), and then re-train it on the same data set, except that this time, we reject the over represented class(es), boost the underrepresented class(es) 
# 

# ## Experiment #: Separate Data into under represented and over represented classes
# 

# 
# ## Experiment #: Restore existing model(s), and train it exclusively on under represented classes, and reject over represented classes during this re-training process. || Then evaluate performance for the complete data set again, and also for a smaller validation data set || Also, check by driving || 
# 

# ## Experiment #: Train against left, right images

# ## Experiment #: Generate Data for under represented classes ( greyscaling, flip, translate or otherwise )
# 

# ## Experiment #: Restore existing models and continue training on under represented classes and compare performance statistically and by driving
# 

# ## Experiment #: Generate descriptive statistics and visualizations with Pandas

# * From the above distribution, we can see that the max number of steering classes exist for steering angle between 0 and -0.1, and that this data is over represented. 
# * This means that we can re-train the data for all cases, except when steering angle is over represented

# In[ ]:

X_train_new = list()
y_train_new = list()

for index, steering_angle in enumerate(y_train):
    steering_angle = float(steering_angle)
    if (steering_angle < -0.1 and steering_angle>-20) or (steering_angle > 0 and steering_angle < 20):
        y_train_new.append(str(steering_angle)) # str needed because for the next step, pandas needs an iterable object
        X_train_new.append(X_train[index])

                
debugLog(y_train_new)


# In[ ]:

new_dataset = pandas.DataFrame({'y_train_new': y_train_new})

new_dataset['y_train_new'] = new_dataset['y_train_new'].astype('float64') 

debugLog(new_dataset.head(20))


# In[ ]:

# histograms

new_dataset.hist()
plt.show()


# In[ ]:

restored_simple_model = load_model('model_simple_model-00-0.54.h5')

debugLog(restored_simple_model)
debugLog(restored_simple_model.layers)


# ## Experiment 1: Comparing Generator approach versus a non-Generator approach in Python

# In[ ]:

# Upper limit of number for this experiment

const_upper_limit = 100


# In[ ]:

# Generator approach
# This can be compared to a recursive approach
# A generator that yields items instead of returning a list

def firstn_generator(n):
    num = 0
    while num < n:
        yield 2*num
        num += 1

before_memory = print_memory_usage()
sum_of_first_n_generator = sum(firstn_generator(const_upper_limit))
after_memory = print_memory_usage()

print("Result ( Generator) : " + str(sum_of_first_n_generator))
print("Change in memory usage ( Kb ) : " + str(after_memory-before_memory))


# In[ ]:

# Non-generator approach
# A traditional / non-recursive approach
# Build and return a complete list

def firstn_nongenerator(n):
        num, nums = 0, []
        while num < n:
            nums.append(num)
            num += 1
        return nums 
    
before_memory = print_memory_usage()
sum_of_first_n_nongenerator = sum(firstn_nongenerator(const_upper_limit))
after_memory = print_memory_usage()

print("Result ( Non Generator approach ) : " + str(sum_of_first_n_nongenerator))
print("Change in memory usage ( Kb ) : " + str(after_memory-before_memory))


# In[ ]:

# Works but no longer in use, hence commented out
# # Use the generator above, to return batches from the training data, and print the size of batch

# debugLog("Started Loading images for generator validation.")

# before_memory = print_memory_usage()

# for image_batch, labels_batch in load_images_generator(training_features_paths_center,training_labels_steering_angle,32):
#     infoLog(" Image Batch Size : " + str(len(image_batch)) + " Label Batch Size : "  + str(len(labels_batch)))
    
# after_memory = print_memory_usage()

# debugLog("Change in memory usage ( Kb ) : " + str(after_memory-before_memory))
# debugLog("Ended Loading images for generator validation.")


# In[ ]:

# Quick experiment to check how to add a value to every element of a numpy array

np_array = np.array([1, 2, 3])
    
debugLog(np_array)

b_array = np_array + .2

debugLog(b_array)


# In[ ]:

# Works but no longer in use, hence commented out
# Convenience method to train a model

# def train_model(param_model, param_model_desc_name, param_train_generator, param_validation_generator, param_samples_per_epoch, param_number_of_validation_samples):
#     # Checkpoint
#     model_filepath = const_model_filename_prefix + model_desc_name + str(get_current_human_readable_time()) + "-{epoch:02d}-{val_acc:.2f}" + const_model_filename_postfix
#     model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     model_callbacks_list = [model_checkpoint]

#     start_time = get_current_time()
#     debugLog(start_time)
#     model_history = param_model.fit_generator(param_train_generator, samples_per_epoch=param_samples_per_epoch, validation_data=param_validation_generator, nb_val_samples=param_number_of_validation_samples, nb_epoch=const_num_epochs, callbacks=model_callbacks_list)
#     debugLog(model_history.history.keys())
#     end_time = get_current_time()
#     debugLog(end_time)
#     print_time_diff(start_time,end_time,"Training " + param_model_desc_name + " for " + str(const_num_epochs) + " epochs")

#     plot_keras_history_object(model_history,param_model_desc_name)
    


# In[ ]:


# Works but no longer in use, hence commented out
# Convenience method to read a CSV file and return left, right, center image paths, and parameters like steering angle, throttle etc.

# def load_csv_file(filename, param_offset = 0.0):

#     debugLog("Started Data Reading :" + filename)
#     training_features_paths_left = list()
#     training_features_paths_center = list()
#     training_features_paths_right = list()
#     training_labels = None
#     training_labels_steering_angle = np.array([])
#     with open(filename, 'r') as csvfile:
#         has_header = csv.Sniffer().has_header(csvfile.read(1024))
#         infoLog("Has Header is " + str(has_header) + " for :" + filename)
#         csvfile.seek(0)  # rewind
#         header = csv.reader(csvfile)
#         if has_header:
#             next(header)  # skip header row
#         reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#         for data_row in reader:
#                 center = data_row[0].strip() # Stripping is important in case there are any unintended leading / trailing spaces in the data
#                 left = data_row[1].strip()
#                 right = data_row[2].strip()
#                 steering = data_row[3]
#                 throttle = data_row[4]
#                 brake = data_row[5]
#                 speed = data_row[6]
                
#                 steering_left = steering + param_offset
#                 steering_right = steering - param_offset
                
#                 # Filtering Logic
#                 steering_angle = float(steering)

#                 # Include all images which are under represented
#                 if (steering_angle < -0.1 and steering_angle > float('-inf') ) or (steering_angle > 0.1 and steering_angle < float('inf')):
#                     training_features_paths_left.append(left)
#                     training_features_paths_center.append(center)
#                     training_features_paths_right.append(right)
#                     training_labels_steering_angle = np.append(training_labels_steering_angle,steering)
            
#                     labels_array = np.array([steering, throttle, brake,speed])
#                     if training_labels is not None:
#                         training_labels = np.vstack((training_labels,labels_array))
#                     else:
#                         training_labels = labels_array
                
#                 # Include 5% of images which are under represented
#                 else:
#                     index = np.random.randint(0, 100)
#                     if index >= 0 and index < 33:
#                         training_features_paths_left.append(left)
#                         training_features_paths_center.append(center)
#                         training_features_paths_right.append(right)
#                         training_labels_steering_angle = np.append(training_labels_steering_angle,steering)
            
#                         labels_array = np.array([steering, throttle, brake,speed])
#                         if training_labels is not None:
#                             training_labels = np.vstack((training_labels,labels_array))
#                         else:
#                             training_labels = labels_array
                        
#     debugLog("Ended Data Reading :" + filename)
#     return training_features_paths_left, training_features_paths_center, training_features_paths_right, training_labels, training_labels_steering_angle


# print_memory_usage()
# training_features_paths_left, training_features_paths_center, training_features_paths_right, training_labels, training_labels_steering_angle = load_csv_file('data/fromudacity/driving_log.csv')
# print_memory_usage()

# # Separate out data into training and validation
# X_train, X_validation, y_train, y_validation = train_test_split(training_features_paths_center, training_labels_steering_angle, test_size=const_validation_data_ratio, random_state=5)

# debugLog(" # Training images : " + str(len(X_train)))
# debugLog(" # Validation images : " + str(len(X_validation)))
# debugLog(" # Training labels : " + str(len(y_train)))
# debugLog(" # Validation labels : " + str(len(y_validation)))


# In[ ]:


# Works but no longer in use, hence commented out
# Convenience method to train a model given a specified CSV file path to the data set
# The model can be a fresh ( randomly initialized ) model, or a model which has been saved and restored from the Disk
# After the completion of training, the best model is saved to disk and also returned

# def train_model_for_csv_file(param_model, param_model_desc_name, csv_file_path, param_epochs = 2, include_center= True, include_left=True, left_offset=0.0, include_right=True, right_offset=0.0 ):
        
#     debugLog(csv_file_path)
    
#     path_prefix = 'data/fromudacity/'
#     # TODO: A bit awkward to separate Udacity Data From Data Generated by me, but works for now
#     if csv_file_path != const_udacity_data_csv_path:
#         path_prefix = None
#         debugLog("path_prefix is None because this is a non-udacity data file.")
#     else:
#         debugLog("path_prefix is : " + csv_file_path)
    
#     # Load and Filter the Data - Only keep the Data we care about ( randomly drop oversampled data ) and return filtered arrays
#     paths_left, paths_center, paths_right, labels_all, labels_steering_angle = load_csv_file(csv_file_path)    
    
#     # Start training process

#     # Define Checkpoint and Early Stopping
#     model_filepath = const_model_filename_prefix + model_desc_name + str(get_current_human_readable_time()) + "-{epoch:02d}-{val_acc:.2f}" + const_model_filename_postfix
#     model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#     early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0)
#     model_callbacks_list = [model_checkpoint, early_stopping]
    
#     # Training on Left Images
#     if include_left == True:
#         debugLog("Training and Validation on left images is enabled.")
#         # Separate out data into training and validation
#         X_train_left, X_validation_left, y_train_left, y_validation_left = train_test_split(paths_left, labels_steering_angle, test_size=const_validation_data_ratio, random_state=5)
#         infoLog(" # Left Training images : " + str(len(X_train_left)))
#         infoLog(" # Left Validation images : " + str(len(X_validation_left)))
#         infoLog(" # Left Training labels : " + str(len(y_train_left)))
#         infoLog(" # Left Validation labels : " + str(len(y_validation_left)))
#         train_generator_left = load_images_generator(X_train_left, y_train_left, 64, path_prefix)
#         validation_generator_left = load_images_generator(X_validation_left, y_validation_left, 64, path_prefix)
#         start_time = get_current_time()
#         debugLog(get_current_human_readable_time())
#         model_history = param_model.fit_generator(train_generator_left, samples_per_epoch = len(X_train_left), validation_data = validation_generator_left, nb_val_samples = len(X_validation_left), nb_epoch = param_epochs, callbacks = model_callbacks_list)
#         debugLog(model_history.history.keys())
#         end_time = get_current_time()
#         debugLog(get_current_human_readable_time())
#         print_time_diff(start_time,end_time,"Training " + param_model_desc_name + " for " + str(param_epochs) + " epochs")    
#          # Generate plot of accuracy over epochs
#         plot_keras_history_object(model_history,param_model_desc_name)
#     else:
#         debugLog("Training and Validation on left images is disabled.")
    
#     # Training on Center Images
#     if include_center == True:
#         debugLog("Training and Validation on center images is enabled.")
#         # Separate out data into training and validation
#         X_train_center, X_validation_center, y_train_center, y_validation_center = train_test_split(paths_center, labels_steering_angle, test_size=const_validation_data_ratio, random_state=5)
#         infoLog(" # Center Training images : " + str(len(X_train_center)))
#         infoLog(" # Center Validation images : " + str(len(X_validation_center)))
#         infoLog(" # Center Training labels : " + str(len(y_train_center)))
#         infoLog(" # Center Validation labels : " + str(len(y_validation_center)))
#         train_generator_center = load_images_generator(X_train_center, y_train_center, 64, path_prefix)
#         validation_generator_center = load_images_generator(X_validation_center, y_validation_center, 64, path_prefix)
#         start_time = get_current_time()
#         debugLog(get_current_human_readable_time())
#         model_history = param_model.fit_generator(train_generator_center, samples_per_epoch = len(X_train_center), validation_data = validation_generator_center, nb_val_samples = len(X_validation_center), nb_epoch = param_epochs, callbacks = model_callbacks_list)
#         debugLog(model_history.history.keys())
#         end_time = get_current_time()
#         debugLog(get_current_human_readable_time())
#         print_time_diff(start_time,end_time,"Training " + param_model_desc_name + " for " + str(param_epochs) + " epochs")    
#          # Generate plot of accuracy over epochs
#         plot_keras_history_object(model_history,param_model_desc_name)
#     else:
#         debugLog("Training and Validation on center images is disabled.")
       
#     # Training on Right Images
#     if include_right == True:
#         debugLog("Training and Validation on right images is enabled.")
#         # Separate out data into training and validation
#         X_train_right, X_validation_right, y_train_right, y_validation_right = train_test_split(paths_right, labels_steering_angle, test_size=const_validation_data_ratio, random_state=5)
#         infoLog(" # Right Training images : " + str(len(X_train_right)))
#         infoLog(" # Right Validation images : " + str(len(X_validation_right)))
#         infoLog(" # Right Training labels : " + str(len(y_train_right)))
#         infoLog(" # Right Validation labels : " + str(len(y_validation_right)))
#         train_generator_right = load_images_generator(X_train_right, y_train_right, 64, path_prefix)
#         validation_generator_right = load_images_generator(X_validation_right, y_validation_right, 64, path_prefix)
#         start_time = get_current_time()
#         debugLog(get_current_human_readable_time())
#         model_history = param_model.fit_generator(train_generator_right, samples_per_epoch = len(X_train_right), validation_data = validation_generator_right, nb_val_samples = len(X_validation_right), nb_epoch = param_epochs, callbacks = model_callbacks_list)
#         debugLog(model_history.history.keys())
#         end_time = get_current_time()
#         debugLog(get_current_human_readable_time())
#         print_time_diff(start_time,end_time,"Training " + param_model_desc_name + " for " + str(param_epochs) + " epochs")    
#          # Generate plot of accuracy over epochs
#         plot_keras_history_object(model_history,param_model_desc_name)
#     else:
#         debugLog("Training and Validation on right images is disabled.")

    
#     # Return the Model
#     return param_model
        

