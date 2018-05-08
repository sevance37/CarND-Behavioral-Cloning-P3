
# coding: utf-8

# In[1]:


# load the needed packages
import csv
import numpy as np
import os
import h5py
import matplotlib
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import tensorflow as tf
tf.python.control_flow_ops = tf

# the Keras routines
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Lambda

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# path variables
on_linux = 1
if (on_linux):
    raw_path = "data_cb"
else:
    raw_path = "C:\\Users\\sevan\\Desktop\\"

hdf5_path = "datahdf5"


# In[3]:


# read in the raw data, store in h5 format for faster read/write
def read_data_csv(data_dir):
    j_clr = [0]      #,1,2]
    shift_clr = [0]  #,0.02,-0.02]
    
    lines = []
    with open(os.path.join(data_dir,"driving_log.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)        
    #lines = lines[0:5]
        
    images = [] 
    angles = []
    
    for (i,line) in enumerate(lines):
        for j in j_clr:
            source_path = line[j]
            if (on_linux):
                source_path = source_path.replace('\\','/')
            (bn,fn) = os.path.split(source_path)
            img_file = os.path.join(data_dir,"IMG",fn)
            if os.path.isfile(img_file):        
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                angles.append(float(line[3]) + shift_clr[j])
            else: 
                print("Missing file {0}".format(img_file))

    X_raw = np.array(images)
    y_raw = np.array(angles)
                
    return X_raw,y_raw

# write the raw data in hdf5
def write_data_hdf5(file_hdf5,X_raw,y_raw):
    hf = h5py.File(file_hdf5, 'w')
    
    hf.create_dataset('X_raw', data=X_raw, compression="gzip")
    hf.create_dataset('y_raw', data=y_raw, compression="gzip")

    hf.close()
    
# read the raw data in hdf5
def read_data_hdf5(file_hdf5):
    hf = h5py.File(file_hdf5, 'r')
 
    X_raw = np.array(hf.get('X_raw'))
    y_raw = np.array(hf.get('y_raw'))

    hf.close()

    return X_raw, y_raw


##### read in the data and store in hdf5 file
X_raw = np.empty((0,160,320,3),dtype=np.uint8)
y_raw = np.empty([0],dtype=np.float)

reread = 1
hdf5_file = os.path.join(hdf5_path,'raw_data.hdf5')
if ((reread) or (not os.path.isfile(hdf5_file))):
    data_dirs = ['data_cb_f1','data_cb_f2','data_cb_b1','data_cb_s1','data_cb_s2']
    for i,data_dir in enumerate(data_dirs):
        X_i, y_i = read_data_csv(os.path.join(raw_path,data_dir))
        print(X_raw.shape)
        print(y_raw.shape)
        X_raw = np.append(X_raw,X_i,axis=0)
        y_raw = np.append(y_raw,y_i,axis=0)

    #write_data_hdf5(hdf5_file,X_raw,y_raw)

#X_raw,y_raw = read_data_hdf5(hdf5_file)


# In[4]:


# The training set will be made up of the images / angles from the simulator
X_train = X_raw
y_train = y_raw

# Increase the amount of training data
# Flip the image and the sign of the angle and add it to the training set.
X_flip = np.empty(X_raw.shape,dtype=X_raw.dtype)
y_flip = np.empty(y_raw.shape,dtype=y_raw.dtype)

fig = plt.figure(num=1,figsize=(12,4))
i_fig = -1 #20

for i in np.arange(len(y_raw)):
    X_flip[i] = cv2.flip(X_raw[i],1)
    y_flip[i] = -y_raw[i]

    if (i == i_fig):
        ax = fig.add_subplot(1,2,1)
        plt.imshow(X_raw[i])
        ax.set_title('Simulation Image', fontsize=20)
        
        ax = fig.add_subplot(1,2,2)
        plt.imshow(X_flip[i])
        ax.set_title('Flipped Image', fontsize=20)
        
        plt.tight_layout()        
        fig.savefig(os.path.join("flipped_image.jpg")) 
        
X_train = np.append(X_raw,X_flip,axis=0)
y_train = np.append(y_raw,y_flip)

print(X_train.shape,y_train.shape)


# In[7]:


# my model based on Nvidia's model
dropout = 0.2

model = Sequential()

# crop and normalize
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) 
    
# the 5 convolutions and 3 fully connect layers
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout))
    
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout))
        
model.add(Convolution2D(48, 4, 4))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout))
    
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(dropout))
    
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
    
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))
    
#nb_epoch=4, batch_size=128 is sufficient
model.compile(loss='mse', optimizer='adam')
hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, 
                 nb_epoch=4, batch_size=128)

model.summary()
print(hist.epoch)
print(hist.history)
    
model.save("model.h5")
    

