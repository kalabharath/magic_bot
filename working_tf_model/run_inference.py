import glob
import sys
import keras
import numpy as np
import tensorflow as tf
from keras_preprocessing import image

if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)

    model = tf.keras.models.load_model(sys.argv[1])

    count = 0
    for tfile in glob.glob('training_dataset_2/target/*.png'):
        img = image.load_img(tfile, target_size=(450, 450))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print (tfile, classes)
        if classes[0] == 1:
            count += 1
    print (count / len(glob.glob('training_dataset_2/target/*.png')))

    count = 0
    for tfile in glob.glob('training_dataset_2/decoy/*.png'):
        img = image.load_img(tfile, target_size=(450, 450))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print (tfile, classes)
        if classes[0] == 0:
            count += 1
    print (count / len(glob.glob('training_dataset_2/decoy/*.png')))