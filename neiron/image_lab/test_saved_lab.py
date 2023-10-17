from keras.models import model_from_json
from IPython.display import Image
from tensorflow.keras.preprocessing import image

print("Загружаю сеть из файлов")
# Загружаем данные об архитектуре сети
json_file = open("CAT_and_DOG.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель
loaded_model = model_from_json(loaded_model_json)
# Загружаем сохраненные веса в модель
loaded_model.load_weights("CAT_and_DOG.h5")
print("Загрузка сети завершена")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

img_path = 'cat.510.jpg'

# Преобразуем картинку в вектор , массив numpy
Image(img_path, width=150, height=150)
import numpy as np
#устанавливаем целевой размер, как и ранее при обучении - 150 на 150
img = image.load_img(img_path, target_size=(150, 150), grayscale=False)
# Преобразуем изображением в массив numpy
x = image.img_to_array(img)
x = 255 - x
x /= 255
x = np.expand_dims(x, axis=0)

prediction = loaded_model.predict(x)
print (prediction)

#bbb = np.argmax(prediction) - это возврат значения индекса наибольшего значения в массиве,  унас всегда одно значение
bbb = np.around(prediction, decimals=0)
print (bbb[0])
print("Номер класса:", bbb)
#print("Название класса:", classes[prediction])