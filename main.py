import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Training CNN/',
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Testing CNN/',
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3),  activation='relu',  input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3),  activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3),  activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128,  activation='relu'),
    Dropout(0.5),
    Dense(2,  activation='softmax')
])

history = model.fit(
    train_generator,
    steps_per_epoch=1,
    validation_data=validation_generator,
    validation_steps=1,
    epochs = 5
)

model.save('/content/drive/My Drive/CNN Model')
img_test = image.load_img("/content/drive/MyDrive/Prediction CNN/Chair Test.jpeg", target_size=(64, 64))
img_test_arr = image.img_to_array(img_test)
img_test_arr = np.expand_dims(img_test_arr, axis=0)

prediction = model.predict(img_test_arr)

class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
predicted_class = class_names[np.argmax(prediction)]
print(predicted_class)
