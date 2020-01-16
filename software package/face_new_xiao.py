import numpy as np
import cv2
import socket
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

rectangleColor = (0,165,255)  

cap = cv2.VideoCapture(0)
cap.set(3,320);
cap.set(4,240);

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    maxArea = 0  
    x = 0  
    y = 0  
    w = 0  
    h = 0

    for (_x,_y,_w,_h) in faces:
        if _w*_h > maxArea:
            x = _x
            y = _y
            w = _w
            h = _h
            maxArea = w*h
    if maxArea > 0:
        cv2.rectangle(img,(x,y),(x+w,y+h),rectangleColor,4)
        a = x+w/2
        b = y+h/2
        print ('face x coord',a)
        print ('face y coord',b)
   
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # cv2.putText(img, "x: {}, y: {}".format(x+(1/2)*w, y+(1/2)*h), (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        # 1, (0, 0, 255), 5)
        cv2.putText(img, "x: {}, y: {}".format(a, b), (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 5)
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()