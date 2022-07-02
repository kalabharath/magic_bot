import os
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_preprocessing import image
import time
try:
    import wandb
    from wandb.keras import WandbCallback
except ImportError:
    os.system('pip install wandb')
    os.system('python tf_hotdog.py &')
    import wandb
    from wandb.keras import WandbCallback
import glob
import numpy as np

if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)

    total_count = 0
    while True:

        wandb.init(project="tf_moon_target_decoy_1")
        wandb.config = {
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 128,
            "dropout": 0.2,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
            "hyper": "parameter"
        }

        # All images will be rescaled by 1/255
        training_dir = 'training_dataset_2/'

        # All images will be rescaled by 1/255
        train_datagen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(directory=training_dir, subset='training',
                                                            class_mode='binary',
                                                            target_size=(450, 450))

        validation_datagen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.2)
        validation_generator = validation_datagen.flow_from_directory(directory=training_dir, subset='training',
                                                                      class_mode='binary', target_size=(450, 450))
        # define the model
        rand_int = np.random.randint(1, 4)
        print("rand_int: ", rand_int)
        # rand_int = 1 works the best
        rand_int = 1
        no_classes = 2
        model = tf.keras.models.Sequential(
            [tf.keras.layers.Conv2D(8 * rand_int, (3, 3), activation='relu', input_shape=(450, 450, 3)),
             tf.keras.layers.MaxPooling2D(2, 2),
             tf.keras.layers.Conv2D(16 * rand_int, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D(2, 2),
             tf.keras.layers.Dropout(0.2 * rand_int),
             tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D(2, 2),
             tf.keras.layers.Dropout(0.2),
             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D(2, 2),
             tf.keras.layers.Conv2D(8 * rand_int, (3, 3), activation='relu'), tf.keras.layers.MaxPooling2D(2, 2),
             tf.keras.layers.Dropout(0.2),
             tf.keras.layers.Flatten(), tf.keras.layers.Dense(512 * rand_int, activation=tf.nn.relu),
             tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
             ])
        checkpoint = ModelCheckpoint(filepath="./ckpts/model.h5", monitor='accuracy', verbose=1, save_best_only=True)
        stopper = EarlyStopping(monitor='accuracy', min_delta=0.003, patience=10, verbose=1, mode='auto')

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000,
        #                                                              decay_rate=0.1)
        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.RMSprop(),
                      metrics=['accuracy'])

        model.fit(train_generator, epochs=wandb.config['epochs'], validation_data=validation_generator, shuffle=True,
                  callbacks=[checkpoint, stopper, WandbCallback()])

        model.save('tf_moon_pattern_' + str(total_count) + '.h5')

        all_classes = []

        count = 0
        for tfile in glob.glob('training_dataset_2/target/*.png'):
            img = image.load_img(tfile, target_size=(450, 450))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images)
            print('target', tfile, classes)
            all_classes.append(classes)
            if int(classes[0]) == 1:
                count += 1
        prob2 = (count / len(glob.glob('training_dataset_2/target/*.png')))

        count = 0

        for tfile in glob.glob('training_dataset_2/decoy/*.png'):
            img = image.load_img(tfile, target_size=(450, 450))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images)
            print ('decoy', tfile, classes)
            all_classes.append(classes)
            if int(classes[0]) == 0:
                count += 1
        prob1 = (count / len(glob.glob('training_dataset_2/decoy/*.png')))

        # print("classification of nmooned and mooned",all_classes)
        if prob1 > 0.8 and prob2 > 0.8:
            print('Model is working', prob1, prob2)
            break
        else:
            print('Model is not working', prob1, prob2)
            total_count += 1
            continue
