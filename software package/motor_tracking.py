import itertools
import time
import serial
import pypot.dynamixel
import numpy as np
import cv2
import socket

ports = pypot.dynamixel.get_available_ports()
print('available ports:', ports)  

if not ports:
    raise IOError('No port available.') 

port = ports[0]
print('Using the first on the list', port)

dxl_io = pypot.dynamixel.DxlIO(port)
print('Connected!')

found_ids = dxl_io.scan(range(3))
print('Found ids:', found_ids)

if len(found_ids) < 2:
    raise IOError('You should connect at least two motors on the bus for this test.')
#chose all motors and enable torque and set the same speed
ids = found_ids[:]
dxl_io.enable_torque(ids)
speed = dict(zip(ids, itertools.repeat(30)))
dxl_io.set_moving_speed(speed)

start_pose=[ -45, 0]
dxl_io.set_goal_position(dict(zip(ids, start_pose)))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

rectangleColor = (0,165,255)  

cap = cv2.VideoCapture(0)
cap.set(3,320);
cap.set(4,240);

k=0.05
xstep=.4*k
ystep=.5*k

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
        if 140<a<180 and 115<b<125:
        	pass
        else:
	        positions=dxl_io.get_present_position(ids)
	        xpos=positions[0]
	        ypos=positions[1]
	        xnew=xpos-(a-160)*xstep
	        ynew=ypos+(b-120)*ystep
	        if -135<xnew<45 and -20<ynew<10:
	        	dxl_io.set_goal_position(dict(zip([1], [xnew])))
	        	dxl_io.set_goal_position(dict(zip([2], [ynew])))

	        # else:
	        # 	dxl_io.set_goal_position(dict(zip([1], [xpos])))
	        # 	dxl_io.set_goal_position(dict(zip([2], [ypos])))
	        print('new',xnew, ynew)

	        time.sleep(.05)
   
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