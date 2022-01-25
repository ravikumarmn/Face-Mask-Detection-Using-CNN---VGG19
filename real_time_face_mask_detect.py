from operator import ne
from keras.models import load_model
import cv2
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

names = {0:"MASK",1:"NO MASK"}
color={1:(0,0,255),0:(0,255,0)}

model = load_model('masknet.h5')
face_clsfr= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)
haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img = capture.read()
    img=cv2.flip(img,1,1)
    img_face = haarcascade.detectMultiScale3(img,minNeighbors = 6,outputRejectLevels = True)
    try:
        faces_probs = sigmoid(img_face[-1])
    except:
        TypeError("Face Not Found")
    for i,d in enumerate(img_face[0]):
        (x,y,w,h) = d
        if faces_probs[i]>0.999:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)

    for i in range(len(img_face[0])):
        if faces_probs[i]>0.95:
            (x,y,w,h) = img_face[0][i]
            crop = img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0
            mask_result = model.predict(crop)
            pos = float(mask_result)
            cv2.putText(img, names[int(np.round(pos))], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color[int(np.round(pos))],2)
            cv2.rectangle(img,(x,y),(x+w,y+h),color[int(np.round(pos))],2)
  
    cv2.imshow('FACE MASK DETECTION',img)
    if cv2.waitKey(2) ==27:
        break
capture.release()
cv2.destroyAllWindows()

