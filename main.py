from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
import scipy
from tensorboard.plugins.hparams import api as hp


train_data_dir='data/train/'
validation_data_dir='data/test/'

# dimensions of our images.
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					horizontal_flip=True,
					fill_mode='nearest')

# Esta es la configuración de aumento que usaremos para la validación
validation_datagen = ImageDataGenerator(rescale=1./255)

# Genera lotes de datos de imágenes de entrenamiento y etiquetas
train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(48, 48),
					batch_size=32,
					class_mode='categorical',
					shuffle=True)

# Genera lotes de datos de imágenes de validación y etiquetas
validation_generator = validation_datagen.flow_from_directory(
					validation_data_dir,
					color_mode='grayscale',
					target_size=(48, 48),
					batch_size=32,
					class_mode='categorical',
					shuffle=True)

# Creamos las etiquetas de clasificación
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

# Creamos el modelo
img, label = train_generator.__next__()

model = Sequential()

# 1 - Capa de entrada
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

# 2 - Segunda capa
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 3 - Tercera capa
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 4 - Cuarta capa
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 5 - Quinta capa
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

 # 6 - Capa de salida
model.add(Dense(7, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


train_path = "data/train/"
test_path = "data/test"

# Asignamos las imágenes de entrenamiento y validación
num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
# Asignamos las imágenes de test
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print(num_train_imgs)
print(num_test_imgs)
epochs=30

# Entrenamiento del modelo
history=model.fit(train_generator,
                steps_per_epoch=num_train_imgs//32,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=num_test_imgs//32)

# Guardamos el modelo
model.save('model_file.h5')