import numpy as np
# importing keras libraries and packages
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

training_dir="data/training"
test_dir = "data/test"
validation_dir = "data/validation"

classifier = Sequential()
classifier.add(Convolution2D(32, (3,3), input_shape=(64, 64, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

train_datagen = image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True)

valid_datagen = image.ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(training_dir, target_size=(64,64),
	batch_size=32, class_mode="binary")

validation_set = valid_datagen.flow_from_directory(validation_dir, target_size=(64,64),
	batch_size=32, class_mode="binary")

classifier.fit_generator(training_set, steps_per_epoch=1000,
	epochs=10, validation_data=validation_set, validation_steps=500)

test_image = image.load_img(test_dir+'/2.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
	prediction = "dog"
else:
	prediction = "cat"

print(prediction)