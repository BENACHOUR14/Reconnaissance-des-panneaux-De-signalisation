# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:56:20 2021

@author: YaSsiN
"""

# import the opencv library
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import time
i = 0
classes = { 1:'Limitation de vitesse (20km/h)',
            2:'Limitation de vitesse (30km/h)', 
            3:'Limitation de vitesse (50km/h)', 
            4:'Limitation de vitesse (60km/h)', 
            5:'Limitation de vitesse (70km/h)', 
            6:'Limitation de vitesse (80km/h)', 
            7:'Fin de la limitation de vitesse (80km/h)', 
            8:'Limitation de vitesse (100km/h)', 
            9:'Limitation de vitesse (120km/h)', 
            10:'Depassement Interdit', 
            11:'Pas de depassement de veh de plus de 3,5 tonnes', 
            12:'Emprise a lintersection', 
            13:'Route prioritaire', 
            14:'Ceder le passage', 
            15:'Stop', 
            16:'Pas de vehicules', 
            17:'Veh > 3,5 tonnes interdit', 
            18:'Entree interdite', 
            19:'Mise en garde generale', 
            20:'Dangereux virage a gauche', 
            21:'Dangereux virage a droite', 
            22:'Double courbe', 
            23:'Route cahoteuse', 
            24:'Route glissante', 
            25:'La route se retrecit a droite', 
            26:'Travaux routiers', 
            27:'Des signaux de trafic', 
            28:'Pietons', 
            29:'Enfants traversant', 
            30:'Traversee de velos', 
            31:'Attention a la glace/neige',
            32:'Traversee danimaux sauvages', 
            33:'Vitesse finale + dépassement des limites', 
            34:'Tournez a droite devant', 
            35:'Tournez a gauche devant', 
            36:'En avant seulement', 
            37:'Allez tout droit ou a droite', 
            38:'Aller tout droit ou a gauche', 
            39:'Restez a droite', 
            40:'Restez a gauche', 
            41:'Rond-point obligatoire', 
            42:'Fin du non-passage', 
            43:'Fin aucun passage veh > 3,5 tonnes' }
  
# définir un objet de capture vidéo
vid = cv2.VideoCapture(0)

WIDTH = vid.get(3)
HEIGHT = vid.get(4)
A = 250
start_point = (int(WIDTH//2 - A//2), int(HEIGHT//2 - A//2))
  
# Coordonnée de fin, ici (220, 220)
# représente le coin inférieur droit du rectangle
end_point = (start_point[0] + A, start_point[1] + A)
  
# Couleur bleue en BGR
color = (255, 0, 0)
  
# Épaisseur de ligne de 2 px
thickness = 2
model = load_model('my_model.h5')
  
while(True):
      
    # Capturer l'image vidéo
    # par image
    ret, frame = vid.read()


    resized = cv2.resize(frame, (30, 30), interpolation = cv2.INTER_AREA)

    # Afficher le cadre résultant
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100,100)
    fontScale              = 1
    fontColor              = (255,0,0)
    lineType               = 2

    #model.predict()
    
    
    
    
    
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
    
    small_image = frame[start_point[1]:start_point[1]+A, start_point[0]:start_point[0]+A]
    small_image = cv2.resize(small_image, (30, 30), interpolation = cv2.INTER_AREA)
    small_image = np.asarray(small_image)
    frame = cv2.flip(frame, 1)

    #pred = model.predict(small_image)
    small_image2 = tf.reshape(small_image, [30, 30, 3])
    X_test=np.array([small_image2])
    pred = model.predict_classes(X_test)[0]
    sign = classes[pred+1]
    
    print(pred)

    cv2.putText(frame,f"{sign}", 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

    cv2.imshow("Frame", frame)
    cv2.imshow('small', small_image)

 
    

      
    # le bouton 'q' est défini comme
    # bouton d'arrêt, vous pouvez utiliser n'importe quel
    # bouton souhaité de votre choix
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(.1)

  
# Après la boucle, relâchez l'objet cap
vid.release()
# Détruire toutes les fenêtres
cv2.destroyAllWindows()