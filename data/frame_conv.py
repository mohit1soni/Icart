import cv2
import numpy as numpy
import os

path= "../../sign_data/video1.mp4"

cap = cv2.VideoCapture(path)
count=0
while(cap.isOpened()):
    ret,frame = cap.read()
    cv2.imwrite("../../sign_data/go_right/image" + str(count)+".jpg",frame)
    count +=1
cap.release()
cv2.destroyAllWindows()