# ИМпорт библиотек для работы с файловой системой
# для операций с файлами и каталогами (копирование, перемещение, создание, удаление)
import shutil
import os

import kwargs as kwargs
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator #функция для загрузки картинок и генератор
from tensorflow.keras.models import Sequential #модели из библиотеке Керас
from tensorflow.keras.layers import Conv2D, MaxPooling2D #слои нейросети
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from os import listdir, sep
from os.path import abspath, basename, isdir
def tree(dir, padding= '  ', print_files=False):
    """
    Эта функция строит дерево поддиректорий и файлов для заданной директории

    Параметры
    ----------
    dir : str
        Path to needed directory
    padding : str
        String that will be placed in print for separating files levels
    print_files : bool
        "Print or not to print" flag
    """
    cmd = "find '%s'" % dir
    files = os.popen(cmd).read().strip().split('\n')
    padding = '|  '
    for file in files:
        level = file.count(os.sep)
        pieces = file.split(os.sep)
        symbol = {0:'', 1:'/'}[isdir(file)]
        if not print_files and symbol != '/':
            continue
        print (padding*level + pieces[-1] + symbol)
def plot_cats_dogs_samples(train_dir, N=4):
  """
    Эта функция строит N самплов каждого класса из датасета Cats vs Dogs

    Параметры
    ----------
    train_dir : str
        Directory with train Cats vs Dogs dataset
    N : int
        Number of samples for each class
  """
  import random
  fig, ax = plt.subplots(2,N,figsize=(5*N,5*2))

  for i,name in enumerate(['cat','dog']):
    filenames = os.listdir(os.path.join(train_dir,name))

    for j in range(N):
      sample = random.choice(filenames)
      image = load_img(os.path.join(train_dir,name,sample))
      ax[i][j].imshow(image)
      ax[i][j].set_xticks([])
      ax[i][j].set_yticks([])
      ax[i][j].set_title(name)
  plt.grid(False)
  plt.show()

# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
#было первоначаль
epochs = 30
#epochs = 3
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 22500
# Количество изображений для проверки
nb_validation_samples = 2500
# Количество изображений для тестирования
nb_test_samples = 2500

base_dir = 'Cat_Dog_data'


train_dir = os.path.join(base_dir, 'train')

test_dir = os.path.join(base_dir, 'test')
tree(base_dir,print_files=False)
plot_cats_dogs_samples(train_dir, N=4)
# X_train, X_test, y_train, y_test = train_test_split(train_dir, test_dir, test_size=0.1, stratify=test_dir, random_state=42)
# Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
# Слой подвыборки, выбор максимального значения из квадрата 2х2
# Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., функция активации ReLU.
# Слой подвыборки, выбор максимального значения из квадрата 2х2
# Слой свертки, размер ядра 3х3, количество карт признаков - 64 шт., функция активации ReLU.
# Слой подвыборки, выбор максимального значения из квадрата 2х2
# flatten - Слой преобразования из двумерного в одномерное представление
# Полносвязный слой, 64 нейрона, функция активации ReLU.
# Слой Dropout.
# Выходной слой, 1 нейрон, функция активации sigmoid
# model = Sequential([
#     Conv2D(32, 3, 32, activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(32, 3, 32, activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, 3, 64, activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# # Компиляция модели
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.evaluate(X_train, y_train, verbose=1)
# # Обучение модели
# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
#
# # Оценка точности модели на тестовых данных
# y_pred = model.predict(test_dir)
# accuracy = accuracy_score(test_dir.argmax(axis=1), y_pred.argmax(axis=1))
# print(f'Accuracy: {accuracy}')
#
# # Сохранение модели в нативном формате Keras
# model.save('image.keras')


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
test_generator = datagen.flow_from_directory(
    #val_dir,
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_validation_samples // batch_size)

# model.summary()
plt.plot(history.history['val_accuracy'], '-o', label='validation accuracy')
plt.plot(history.history['accuracy'], '--s', label='training accuracy')
plt.legend();

plt.plot(history.history['val_accuracy'], '-o', label='validation accuracy')
plt.plot(history.history['accuracy'], '--s', label='training accuracy')
plt.legend();

plt.plot(history.history['val_loss'], '-o', label='validation loss')
plt.plot(history.history['loss'], '--s', label='training loss')
plt.legend();

model.evaluate(test_generator)

print("Сохраняем сеть")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("CAT_and_DOG.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("CAT_and_DOG.h5")
print("Сохранение сети завершено")