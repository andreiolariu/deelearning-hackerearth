import numpy as np
np.random.seed(1337)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from util import load_data_full

from imgaug import augmenters as iaa

from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train) = load_data_full()

BATCH_SIZE = 32

def create_model():
    base_model = applications.ResNet50(weights='imagenet', include_top=False, \
                    input_shape=x_train.shape[1:])
    add_model = Sequential()
    add_model.add(BatchNormalization(input_shape=base_model.output_shape[1:]))
    add_model.add(Flatten())
    add_model.add(Dense(384, activation='relu'))
    add_model.add(BatchNormalization())
    add_model.add(Dropout(0.5))
    add_model.add(Dense(np.max(y_train) + 1, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), \
                      metrics=['accuracy'])
    for layer in model.layers[:-1]:
        layer.trainable = False
        
    return model


def image_augment(i):
    global seq
    images = np.expand_dims(i, 0)
    images = seq.augment_images(images)
    return images[0]

def image_augment2(i):
    global seq
    c_i = i.copy()
    images = np.expand_dims(c_i, 0)
    images *= 255
    images = seq.augment_images(images)
    images = np.true_divide(images, 255)
    return images[0]

def train_model(epochs, train_generator):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
        epochs=epochs,
    )

model = create_model()

train_datagen = ImageDataGenerator()
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(10, train_generator)

train_datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True
)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(10, train_generator)

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(10, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths1.model')

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(15, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths2.model')

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(10, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths3.model')

train_model(30, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths4.model')

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
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(30, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths5.model')

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
      scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
      shear=(-40, 40),
      rotate=(-45, 45),
      order=[0, 1],
      translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
      mode='reflect',
    ),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(30, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths6.model')

seq = iaa.Sequential([
    iaa.Sometimes(0.75, [iaa.OneOf([
        iaa.Add((-100, 100)),
        iaa.Multiply((0.15, 1.8)),
        iaa.ContrastNormalization((0.15, 2))
    ])]),
    iaa.Sometimes(0.1, [iaa.CoarseDropout((0.05, 0.2), size_percent=(0.05,0.15))]),
    iaa.Sometimes(0.7, [iaa.SomeOf((1, None), [
        iaa.PiecewiseAffine(scale=(0.01, 0.04), mode='reflect'),
        iaa.OneOf([
            iaa.AverageBlur(k=((3, 7), 1.01)),
            iaa.AverageBlur(k=(1.01, (3, 7)))
        ])
    ], random_order=True)]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.75, [iaa.Affine(
      scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
      shear=(-35, 35),
      rotate=(-45, 45),
      order=[0, 1],
      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
      mode='reflect',
    )]),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(50, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths7.model')

seq = iaa.Sequential([
    iaa.Sometimes(0.75, [iaa.OneOf([
        iaa.Add((-75, 75)),
        iaa.Multiply((0.3, 1.7)),
        iaa.ContrastNormalization((0.5, 1.5))
    ])]),
    iaa.Sometimes(0.7, [iaa.SomeOf((1, None), [
        iaa.PiecewiseAffine(scale=(0.01, 0.02), mode='reflect'),
        iaa.OneOf([
            iaa.AverageBlur(k=((3, 7), 1.01)),
            iaa.AverageBlur(k=(1.01, (3, 7)))
        ])
    ], random_order=True)]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5, [iaa.OneOf([iaa.Affine(
      scale=(0.7, 1.3),
      shear=(-10, 10),
      rotate=(-180, 180),
      order=[0, 1],
      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
      mode=m
    ) for m in ['constant', 'reflect', 'wrap', 'symmetric']])]),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(30, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths8.model')

seq = iaa.Sequential([
    iaa.Sometimes(0.75, [iaa.OneOf([
        iaa.Add((-30, 30)),
        iaa.Multiply((0.7, 1.3)),
        iaa.ContrastNormalization((0.8, 1.2))
    ])]),
    iaa.Sometimes(0.7, [iaa.SomeOf((1, None), [
        iaa.PiecewiseAffine(scale=(0.01, 0.02), mode='reflect'),
        iaa.OneOf([
            iaa.AverageBlur(k=((3, 7), 1.01)),
            iaa.AverageBlur(k=(1.01, (3, 7)))
        ])
    ], random_order=True)]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5, [iaa.OneOf([iaa.Affine(
      scale=(0.9, 1.2),
      rotate=(-180, 180),
      order=[0, 1],
      mode=m
    ) for m in ['constant', 'reflect', 'wrap', 'symmetric']])]),
])

train_datagen = ImageDataGenerator(preprocessing_function=image_augment2)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

train_model(30, train_generator)

model.save('hodor_resnets/resnet50_augmentation_experiment_ths9.model')