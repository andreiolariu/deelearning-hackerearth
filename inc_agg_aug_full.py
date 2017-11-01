import numpy as np
np.random.seed(1337)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import pickle
import gc

import imgaug as ia
from imgaug import augmenters as iaa
import keras
from keras import applications
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split

from util import load_data_local_eval, load_data_full

VAL_SPLIT = 0.2
BATCH_SIZE = 23
checkpoint = 1

MODEL_NAME = 'inc_agg_aug_full'
# x_train, x_val, y_train, y_val = load_data_local_eval(VAL_SPLIT)
x_train, y_train = load_data_full()

def save_model():
    global checkpoint
    cp_name = '%s_cp%s.model' % (MODEL_NAME, checkpoint)
    model.save(cp_name)
    print('saved %s' % cp_name)
    checkpoint += 1


base_model = applications.InceptionResNetV2( \
    weights='imagenet', include_top=False, input_shape=x_train.shape[1:])

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(BatchNormalization())
add_model.add(Dense(256, activation='relu'))
add_model.add(BatchNormalization())
add_model.add(Dropout(0.5))
add_model.add(Dense(np.max(y_train) + 1, activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), \
                  metrics=['accuracy'])

base_model.trainable = False

train_datagen = ImageDataGenerator()

def train_model(train_datagen, epochs):
    train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
        epochs=epochs,
    )
    return history

train_model(train_datagen, 5)
save_model() # cp1
train_datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
)
train_model(train_datagen, 20)

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
)
train_model(train_datagen, 10)
save_model() # cp2

train_datagen = ImageDataGenerator(
        rotation_range=40, 
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
)
train_model(train_datagen, 10)
save_model() # cp3

train_datagen = ImageDataGenerator(
        rotation_range=90, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
)
train_model(train_datagen, 60)
save_model() # cp4

train_datagen = ImageDataGenerator(
        rotation_range=90, 
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.4,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
)
train_model(train_datagen, 20)
save_model() # cp5

ia.seed(1)
sometimes = lambda aug: iaa.Sometimes(0.3, aug)

seq = iaa.Sequential([
  iaa.Fliplr(0.5),
  iaa.Fliplr(0.5),
  iaa.Affine(
      scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
      translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
      rotate=(-60, 60),
      shear=(-30, 30),
      order=[0, 1],
      mode='reflect',
  ),
])

def image_augment(i):
  images = np.expand_dims(i, 0)
  images = seq.augment_images(images)
  return images[0]

train_datagen = ImageDataGenerator(preprocessing_function=image_augment)
train_model(train_datagen, 20)
save_model() # cp6

seq = iaa.Sequential([
  iaa.Fliplr(0.5),
  iaa.Fliplr(0.5),
  iaa.Affine(
      scale={"x": (0.6, 1.5), "y": (0.6, 1.5)},
      translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
      rotate=(-90, 90),
      shear=(-40, 40),
      order=[0, 1],
      mode='reflect',
  ),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment)
train_model(train_datagen, 20)
save_model() # cp7

seq = iaa.Sequential([
  iaa.Fliplr(0.5),
  iaa.Fliplr(0.5),
  iaa.Affine(
      scale={"x": (0.5, 1.7), "y": (0.5, 1.7)},
      translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
      rotate=(-90, 90),
      shear=(-50, 50),
      order=[0, 1],
      mode='reflect',
  ),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment)
train_model(train_datagen, 30)
save_model() # cp8

seq = iaa.Sequential([
  iaa.Fliplr(0.5),
  iaa.Fliplr(0.5),
  iaa.Affine(
      scale={"x": (0.45, 1.9), "y": (0.45, 1.9)},
      translate_percent={"x": (-0.55, 0.55), "y": (-0.55, 0.55)},
      rotate=(-90, 90),
      shear=(-60, 60),
      order=[0, 1],
      mode='reflect',
  ),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment)
train_model(train_datagen, 30)
save_model() # cp9

seq = iaa.Sequential([
  iaa.Fliplr(0.5),
  iaa.Fliplr(0.5),
  iaa.Affine(
      scale={"x": (0.4, 2), "y": (0.4, 2)},
      translate_percent={"x": (-0.6, 0.6), "y": (-0.6, 0.6)},
      rotate=(-90, 90),
      shear=(-65, 65),
      order=[0, 1],
      mode='reflect',
  ),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment)
train_model(train_datagen, 30)
save_model() # cp10

seq = iaa.Sequential([
  iaa.Fliplr(0.5),
  iaa.Fliplr(0.5),
  iaa.Affine(
      scale={"x": (0.35, 2.1), "y": (0.35, 2.1)},
      translate_percent={"x": (-0.65, 0.65), "y": (-0.65, 0.65)},
      rotate=(-90, 90),
      shear=(-70, 70),
      order=[0, 1],
      mode='reflect',
  ),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment)
train_model(train_datagen, 30)
save_model() # cp11

train_model(train_datagen, 40)
save_model() # cp12