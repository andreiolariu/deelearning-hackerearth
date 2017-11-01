import numpy as np
np.random.seed(1337)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
checkpoint = 13

MODEL_NAME = 'inc_agg_aug_full'
# x_train, x_val, y_train, y_val = load_data_local_eval(VAL_SPLIT)
x_train, y_train = load_data_full()

def save_model():
    global checkpoint
    cp_name = '%s_cp%s.model' % (MODEL_NAME, checkpoint)
    model.save(cp_name)
    print('saved %s' % cp_name)
    checkpoint += 1

model = keras.models.load_model('inc_agg_aug_full_cp12.model')

train_datagen = ImageDataGenerator()

val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)

def train_model(train_datagen, epochs):
    train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
        epochs=epochs,
        # validation_data=val_generator,
        # validation_steps=x_val.shape[0] // BATCH_SIZE
    )
    return history


def image_augment2(i):
    global seq
    c_i = i.copy()
    images = np.expand_dims(c_i, 0)
    images *= 255
    images = seq.augment_images(images)
    images = np.true_divide(images, 255)
    return images[0]

seq = iaa.Sequential([
    iaa.Sometimes(0.7, [iaa.Add((-80, 80))]),
    iaa.Sometimes(0.3, [iaa.CoarseDropout((0.05, 0.2), size_percent=0.1)]),
    iaa.Sometimes(0.7, [iaa.SomeOf((1, None), [
        iaa.PiecewiseAffine(scale=(0.01, 0.04), mode='reflect'),
        iaa.OneOf([
            iaa.AverageBlur(k=((3, 7), 1.01)),
            iaa.AverageBlur(k=(1.01, (3, 7)))
        ])
    ], random_order=True)]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(
        scale={"x": (0.35, 2.1), "y": (0.35, 2.1)},
        translate_percent={"x": (-0.65, 0.65), "y": (-0.65, 0.65)},
        rotate=(-90, 90),
        shear=(-70, 70),
        order=[0, 1],
        mode='reflect',
    ),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_model(train_datagen, 30)
save_model() # cp13

for layer in model.layers:
    layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_model(train_datagen, 30)
save_model() # cp14

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_model(train_datagen, 30)
save_model() # cp15

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_model(train_datagen, 30)
save_model() # cp16

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_model(train_datagen, 30)
save_model() # cp17

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_model(train_datagen, 30)
save_model() # cp18

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_model(train_datagen, 30)
save_model() # cp19