# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:14:14 2021

@author: YaSsiN
"""
import numpy as np 
import pandas as pd 
import tensorflow as tf
from PIL import Image
import os
#train_test_split pour diviser les données en deux dataframes formation et test
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
data = []
labels = []
classes = 43
chemin_courant = os.getcwd()
#Récupérer les images et leurs étiquettes
for i in range(classes):
    chemin = os.path.join(chemin_courant,'train',str(i))
    images = os.listdir(chemin)
    for a in images:
        try:
            image = Image.open(chemin + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Erreur de chargement de l'image")
#Conversion de listes en tableaux numpy
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
X_train, X_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


y_train =to_categorical(y_train, 43)
y_test =to_categorical(y_test, 43)


#Construire le modèle
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
historique = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))



#tester la précision sur l'ensemble de données de test
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    image = np.array(image)
    data.append(image)

X_test=np.array(data)

prediction = model.predict_classes(X_test)

#Accuracy with the test data

print(accuracy_score(labels, prediction))

model.save("my_model.h5")


