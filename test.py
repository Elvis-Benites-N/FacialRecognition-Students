import cv2
import numpy as np
from keras.models import load_model

# Cargamos el modelo
model=load_model('model_file_30epochs.h5')

# Cargamos la imagen
video=cv2.VideoCapture(0)

# Creamos el clasificador de rostros
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# Creamos un bucle para que el programa no se cierre
while True:
    # Capturamos el frame
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    # Dibujamos un rectángulo alrededor de la cara
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        # Redimensionamos la imagen
        resized=cv2.resize(sub_face_img,(48,48))
        # Normalizamos la imagen
        normalize=resized/255.0
        # Cambiamos la forma de la imagen
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        # Hacemos la predicción
        result=model.predict(reshaped)
        # Obtenemos la etiqueta
        label=np.argmax(result, axis=1)[0]
        # Dibujamos el rectángulo
        print(label)
        # Dibujamos el rectángulo
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        # Escribimos la etiqueta
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    # Mostramos el frame
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
# Liberamos la cámara y cerramos todas las ventanas
video.release()
cv2.destroyAllWindows()