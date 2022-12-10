```python
import tensorflow as tf

import tensorflow_datasets as tfds
```


```python
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
```


```python
from sklearn import metrics
from sklearn.utils import shuffle
```


```python
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
```


```python
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from numpy import *
from PIL import Image
import theano
```


```python
path_test = "C:/Users/User/Room_dataset/"
```


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.callbacks import EarlyStopping
```


```python
dir_list = sorted(os.listdir('C:/Users/User/Room_dataset'))
class_names_lst = [dir_name for dir_name in dir_list]
class_names_lst
```


```python
g_generator = ImageDataGenerator(
    rescale = 1. / 255,
    horizontal_flip = True,
    rotation_range = 20,
    shear_range = 0.2,
    fill_mode = 'nearest',
    validation_split=0.1
)
train_data = img_generator.flow_from_directory(
    'C:/Users/User/Room_dataset',
    target_size = (255, 255),
    color_mode = 'rgb',
    classes = class_names_lst,
    class_mode = 'categorical',
    batch_size  = 64,
    subset = 'training'
)
valid_data = img_generator.flow_from_directory(
    'C:/Users/User/Room_dataset',
    target_size = (255, 255),
    color_mode = 'rgb',
    classes = class_names_lst,
    class_mode = 'categorical',
    batch_size  = 64,
    subset = 'validation'
)

```


```python
base2_model = VGG19(
    input_shape=(255, 255, 3),
    include_top=False,
    weights="imagenet"
)
```


```python
early_stop=EarlyStopping(monitor='val_loss',patience=5, restore_best_weights=True)
```


```python
model = keras.Sequential([
    base2_model,
    layers.Flatten(),
    layers.Dense(5, activation = 'softmax')
])
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = 'categorical_crossentropy',
    metrics = 'accuracy'
)
model.summary()
```


```python
model.fit(
    train_data,
    epochs = 7,
    batch_size = 64,
    validation_data = valid_data,
    callbacks = [early_stop]
)
```
