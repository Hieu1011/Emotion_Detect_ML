import os
import pandas as pd
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

# Data filepath 
train_dir = 'data/train'
val_dir = 'data/test'

# Preprocessing
train_data_gen = ImageDataGenerator(rescale=1./255)
valid_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')
validation_generator = valid_data_gen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Build model 
def build_model(model):
    # Build CNN Model (10 layers)
    # module 1
    model.add(Conv2D(2*2*64, kernel_size=(3, 3), input_shape=(48, 48, 1), data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(2*2*64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # module 2
    model.add(Conv2D(2*64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(2*64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # module 3
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # flatten
    model.add(Flatten())

    # dense 1
    model.add(Dense(2*2*2*64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # dense 2
    model.add(Dense(2*2*64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # dense 3
    model.add(Dense(2*64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #output layer
    model.add(Dense(7, activation='softmax'))

    return model
emotion_model = Sequential()
build_model(emotion_model)

emotion_dict = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Train model
es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), metrics=['accuracy'])
# emotion_model_info = emotion_model.fit_generator(
#                             train_generator,
#                             steps_per_epoch=28709 / 64,
#                             epochs=50,
#                             callbacks = [es],
#                             validation_data=validation_generator)
# emotion_model.save_weights('model/best.h5')

# Load weights
emotion_model.load_weights('model/best.h5')

cv2.ocl.setUseOpenCL(False)

# Start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
