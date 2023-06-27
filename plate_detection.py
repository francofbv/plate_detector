import cv2
import numpy as np


frameWidth = 640
frameHeight = 480
color = (255,0,255)
plateCascade = cv2.CascadeClassifier('/home/francobv/Downloads/haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4, frameHeight)
cap.set(10,50)
while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    number_plates = plateCascade.detectMultiScale(imgGray,1.1,4)

    for (x,y,w,h) in number_plates:
        area = w*h
        if area > 500:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,'NumberPlate',(x,y-y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow('ROI', imgRoi)

    cv2.imshow('result',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
