import pickle

import imgaug as ia
from imgaug import augmenters as iaa
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from tqdm import tqdm


VAL_SPLIT = 0.2
BATCH_SIZE = 32
DATA_FOLDER = '../data'


def generate_submission(model_name, local_eval=False, sub_name=None):
    '''
        Given a model name (without the '.model' in the end), this
        loads it, evaluates it on 20% of training data (optional),
        and generates a submission file (saved in a csv with the
        model name).
    '''
    if model_name.endswith('.model'):
        model_name = model_name[:-6]
    model = load_model('%s.model' % model_name)
    
    if local_eval:
        with open('%s/labeled_data.pickle' % DATA_FOLDER, 'rb') as f:
            (x_train, y_train) = pickle.load(f)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
                test_size=VAL_SPLIT, random_state=42)
        predictions = model.predict(x_val)
        predictions = np.argmax(predictions, axis= 1)
        print(classification_report(y_val, predictions))

    with open('%s/test_data.pickle' % DATA_FOLDER, 'rb') as f:
        (ids, x_test) = pickle.load(f)
    with open('%s/id_to_label.pickle' % DATA_FOLDER, 'rb') as f:
        id_to_label = pickle.load(f)

    # Get test labels ids
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)

    pred_labels = [id_to_label[k] for k in predictions]

    sub = pd.DataFrame({'image_id':ids, 'label':pred_labels})
    if not sub_name:
        sub_name = '%s.csv' % model_name
    sub.to_csv(sub_name, index=False)


ia.seed(1)
sometimes = lambda aug: iaa.Sometimes(0.3, aug)
seq1 = iaa.Sequential([
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
seq2 = iaa.Sequential([
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

def _image_augment(i, seq):
    images = np.expand_dims(i, 0)
    images = seq.augment_images(images)
    return images[0]

def _image_augment_v1(i):
    return _image_augment(i, seq1)

def _image_augment_v2(i):
    return _image_augment(i, seq2)

test_datagen_list = [
    ImageDataGenerator(
        rotation_range=45, 
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    ),
    ImageDataGenerator(preprocessing_function=_image_augment_v1),
    ImageDataGenerator(preprocessing_function=_image_augment_v2),
]


def generate_submission_augmented(model_name, aug_version, sub_name=None):
    '''
        Given a model name (without the '.model' in the end), this
        loads it and generates a submission file (saved in a csv with the
        model name) by averaging predictions for 10 augmented images for 
        each test image.

        Need to also specify aug_version:
            0 - the gentle one used by Hodor
            1 - the aggressive one used by Andrei
            2 - a little more aggressive
    '''
    if model_name.endswith('.model'):
        model_name = model_name[:-6]
    model = load_model('%s.model' % model_name)

    with open('%s/test_data.pickle' % DATA_FOLDER, 'rb') as f:
        (ids, x_test) = pickle.load(f)
    with open('%s/id_to_label.pickle' % DATA_FOLDER, 'rb') as f:
        id_to_label = pickle.load(f)

    test_datagen = test_datagen_list[aug_version]
    test_datagen.fit(x_test)
    test_generator = test_datagen.flow(x_test, range(len(x_test)), \
            batch_size=len(x_test))

    d = dict()
    num_copies = 10
    for i in tqdm(range(num_copies)):
        t = test_generator.next()
        images, arb_ids = t
        predictions = model.predict(images)
        for p, l in zip(predictions, arb_ids):
            if l not in d:
                d[l] = []
            d[l].append(p)

    for k in list(d.keys()):
        d[k] = np.array(d[k])

    predictions = []
    for im_index in sorted(list(d.keys())):
        predictions.append(np.argmax(np.mean(d[im_index], axis=0)))

    pred_labels = [id_to_label[k] for k in predictions]

    sub = pd.DataFrame({'image_id':ids, 'label':pred_labels})
    if not sub_name:
        sub_name = '%s_test_aug.csv' % model_name
    sub.to_csv(sub_name, index=False)


def load_data_local_eval(val_split):
    with open('%s/labeled_data.pickle' % DATA_FOLDER, 'rb') as f:
        (x_train, y_train) = pickle.load(f)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, \
        test_size=val_split, random_state=42)
    return (x_train, x_val, y_train, y_val)


def load_data_full():
    with open('%s/labeled_data.pickle' % DATA_FOLDER, 'rb') as f:
        (x_train, y_train) = pickle.load(f)
    return (x_train, y_train)


def generate_submission_ensemble(name, models):
    '''
    Generate a submission from an ensemble of models.

    Please provide a list of triplets of (model_name, weight, aug_version).
    If you don't want augmentation for a given model, pass aug_version=None.
    If you don't wan't to remap keys, pass None
    '''
    with open('%s/test_data.pickle' % DATA_FOLDER, 'rb') as f:
        (ids, _) = pickle.load(f)
    with open('%s/id_to_label.pickle' % DATA_FOLDER, 'rb') as f:
        id_to_label = pickle.load(f)
    label_to_id = {v:k for k, v in id_to_label.items()}

    prediction_list = []
    weights = []
    for model_name, weight, aug_version in models:
        weights.append(weight)
        sub_name = 'ensemble_%s_%s%s.csv' % (
            name,
            model_name,
            '_aug%s' % aug_version if aug_version is not None else ''
        )
        try:
            pred_data = pd.read_csv(sub_name)
        except:
            print ("Couldn't find saved output data for {}".format(model_name))
            if aug_version is None:
                generate_submission(model_name, sub_name=sub_name)
            else:
                generate_submission_augmented(model_name, \
                    aug_version=aug_version, sub_name=sub_name)
            pred_data = pd.read_csv(sub_name)

        pred_dict = {}
        for _, row in pred_data.iterrows():
            pred_dict[row[0]] = label_to_id[row[1]]
        prediction = np.array([pred_dict[i] for i in ids])
        prediction_list.append(prediction)

    weights = np.array(weights)
    predictions = np.array(prediction_list)
    # convert to one hot encoding
    predictions = (np.arange(predictions.max() + 1) == \
            predictions[:,:,None]).astype(int)

    prediction = np.argmax(np.sum(predictions * \
        weights[:, None, None], axis=0), axis=1)

    pred_labels = [id_to_label[k] for k in prediction]

    sub = pd.DataFrame({'image_id':ids, 'label':pred_labels})
    sub.to_csv('ensemble_%s.csv' % name, index=False)