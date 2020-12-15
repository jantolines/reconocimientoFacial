import cv2
import os
import numpy as np

dataPath = r'E:\UNIR\SEGUNDO CUATRIMESTRE\PRACTICA EMPRESARIAL\Desarrollo Proyecto\Proyecto2\data'
peopleList=os.listdir(dataPath)
print('Lista de personas: ' + str(peopleList))

labels=[]
facesData=[]
label = 0

for nameDir in peopleList:
    personPath = dataPath + '\\' + nameDir
    print ('Leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '\\' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '\\'+ fileName,0))
        image= cv2.imread(personPath + '\\'+ fileName,0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)
    label=label+1

print ('labels= ', str(labels))
print('Numero de etiquetas 0= ', np.count_nonzero(np.array(labels)==0))
print('Numero de etiquetas 1= ', np.count_nonzero(np.array(labels)==1))
print('Numero de etiquetas 2= ', np.count_nonzero(np.array(labels)==2))
print('Numero de etiquetas 3= ', np.count_nonzero(np.array(labels)==3))
print('Numero de etiquetas 4= ', np.count_nonzero(np.array(labels)==4))


#METODO EIGEN FACES
face_recognizerEigen=cv2.face.EigenFaceRecognizer_create()
print('Entrenando Eigen...')
face_recognizerEigen.train(facesData,np.array(labels))

#Almacenando modelo obtenido
face_recognizerEigen.write('modeloEigenFaceJRAE.xml')
print("Modelo Eigen almacenado...")

#METODO FISHER FACES
face_recognizerFisher=cv2.face.FisherFaceRecognizer_create()
print('Entrenando Fisher...')
face_recognizerFisher.train(facesData,np.array(labels))

#Almacenando modelo obtenido
face_recognizerFisher.write('modeloFisherFaceJRAE.xml')
print("Modelo Fisher almacenado...")

#METODO LBPH FACES
face_recognizerLBPH=cv2.face.LBPHFaceRecognizer_create()
print('Entrenando LBPH...')
face_recognizerLBPH.train(facesData,np.array(labels))

#Almacenando modelo obtenido
face_recognizerLBPH.write('modeloLBPHFaceJRAE.xml')
print("Modelo LBPH almacenado...")