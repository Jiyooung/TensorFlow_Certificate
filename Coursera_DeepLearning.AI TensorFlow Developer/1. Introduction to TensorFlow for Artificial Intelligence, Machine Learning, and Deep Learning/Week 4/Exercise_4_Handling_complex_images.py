import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

data_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip'
# 데이터 다운로드, 처음 한 번만 수행
# import requests
# req = requests.get(data_url)
# # print(req.status_code) # 200 성공
# with open("happy-or-sad.zip", "wb") as code:
#     code.write(req.content)

# url 파일 다운로드 또 다른 방법
# import urllib.request
# urllib.request.urlretrieve(data_url, "happy-or-sad.zip")

# # 파일 압축 해제, 처음 한 번만 수행
# path = f"{getcwd()}/happy-or-sad.zip"
# zip_ref = zipfile.ZipFile(path, 'r')
# zip_ref.extractall("/tmp/h-or-s")
# zip_ref.close()

# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
         # Your Code
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > DESIRED_ACCURACY):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. 
    # Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        "/tmp/h-or-s",
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')
        
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks=[callbacks]
    )

    # model fitting
    return history.history['accuracy'][-1]

# The Expected output: "Reached 99.9% accuracy so cancelling training!""
train_happy_sad_model()








