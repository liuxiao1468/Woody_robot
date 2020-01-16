import cv2
import glob
import random
import dlib
import numpy as np
import math
import itertools
from sklearn.svm import SVC
import PIL
from PIL import Image
from sklearn.externals import joblib
import time
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import itertools
import serial
import pypot.dynamixel
import socket
from threading import Thread

def get_datasets(emotion):
	files = glob.glob("/home/leo/woody_vision/sorted_CK+//%s//*" %emotion)
	random.shuffle(files)
	training = files[:int(len(files)*0.7)] #get first 80% of file list
	prediction = files[-int(len(files)*0.3):] #get last 20% of file list
	return training, prediction




def get_landmarks(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)
	detections = detector(clahe_image, 1)
	for k,d in enumerate(detections): #For all detected face instances individually
		shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
		xlist = []
		ylist = []
		landmarks= []
		for i in range(0,68): #Store X and Y coordinates in two lists
			cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 
			#For each point, draw a red circle with thickness2 on the original frame
			xlist.append(float(shape.part(i).x))
			ylist.append(float(shape.part(i).y))

		xmean = np.mean(xlist) #Find both coordinates of centre of gravity
		ymean = np.mean(ylist)
		x_max = np.max(xlist)
		x_min = np.min(xlist)
		y_max = np.max(ylist)
		y_min = np.min(ylist)
		cv2.rectangle(clahe_image,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,150,0),2)
		# print ("centre of gravity",xmean, ymean)
		# print ("range of the face",x_max, x_min, y_max, y_min)
		cv2.circle(clahe_image, (int(xmean), int(ymean) ), 1, (0,255,255), thickness=2) 
		x_start = int(x_min-5)
		y_start = int(y_min-((ymean - y_min)/3))
		w = int(x_max+5) - x_start
		h = int(y_max+5) - y_start

		xlist[:] = [x-x_start for x in xlist]
		ylist[:] = [y-y_start for y in ylist]

		xlist = np.array(xlist,dtype = np.float64)
		ylist = np.array(ylist,dtype = np.float64)
		# xlist = np.float32(xlist)
		# ylist = np.float32(ylist)

	if len(detections) > 0:
		return xlist, ylist
	else: #If no faces are detected, return error message to other function to handle
		xlist = np.array([])
		ylist = np.array([])
		return xlist, ylist



# def linear_interpolation(xlist,ylist):
#     xlist = np.array(xlist,dtype = np.float64)
#     ylist = np.array(ylist,dtype = np.float64)
#     x_new = np.array([])
#     y_new = np.array([])
#     for i in range (len(xlist)-1):
#         x_new = np.concatenate((x_new,[(xlist[i]+xlist[i+1])/2.0]))
#         y_new = np.concatenate((y_new,[(ylist[i]+ylist[i+1])/2.0]))
#     xlist = np.append(xlist, x_new)
#     ylist = np.append(ylist, y_new)
#     return xlist, ylist

def wave():
	waveids = [3,4,5,6,7,8,9,10,11,12]
	print('attempting')
	speed = dict(zip(waveids, itertools.repeat(40)))
	dxl_io.set_moving_speed(speed)
	start_pose=[45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]
	dxl_io.set_goal_position(dict(zip(waveids, start_pose)))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip(waveids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(55)))) #this still needs to change
	dxl_io.set_goal_position(dict(zip([3], [-70])))
	time.sleep(3)
	dxl_io.set_moving_speed(dict(zip([5], itertools.repeat(65))))
	dxl_io.set_goal_position(dict(zip([5], [-36])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-7.5])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-36])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-7.5])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-36])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-7.5])))
	time.sleep(0.5)
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3], [32.7])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([3], [40])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip(waveids, start_pose)))
	time.sleep(2)

def waveleft():
	waveids = [3,4,5,6,7,8,9,10,11,12]
	print('attempting')
	speed = dict(zip(waveids, itertools.repeat(40)))
	dxl_io.set_moving_speed(speed)
	start_pose=[45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]
	dxl_io.set_goal_position(dict(zip(waveids, start_pose)))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip(waveids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([8], itertools.repeat(55)))) #this still needs to change
	dxl_io.set_goal_position(dict(zip([8], [70])))
	time.sleep(3)
	dxl_io.set_moving_speed(dict(zip([10], itertools.repeat(65))))
	dxl_io.set_goal_position(dict(zip([10], [36])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [7.5])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [36])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [7.5])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [36])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [7.5])))
	time.sleep(0.5)
	dxl_io.set_moving_speed(dict(zip([8], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([8], [-32.7])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([8], [-40])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip(waveids, start_pose)))
	time.sleep(2)

def linear_interpolation(xlist,ylist):
	xlist = np.array(xlist,dtype = np.float64)
	ylist = np.array(ylist,dtype = np.float64)
	x_new = np.array([])
	y_new = np.array([])
	x = np.array([])
	y = np.array([])
	for i in range (len(xlist)-1):
		x_new = np.concatenate((x_new,[(xlist[i]+xlist[i+1])/2.0]))
		y_new = np.concatenate((y_new,[(ylist[i]+ylist[i+1])/2.0]))

	for j in range (len(xlist)):
		if j<(len(xlist)-1):
			x = np.concatenate((x,[xlist[j]]))
			x = np.concatenate((x,[x_new[j]]))
			y = np.concatenate((y,[ylist[j]]))
			y = np.concatenate((y,[y_new[j]]))
		else:
			x = np.concatenate((x,[xlist[j]]))
			y = np.concatenate((y,[ylist[j]]))
	return x, y


def extract_AU(xlist,ylist):
	AU_feature = []
	Norm_AU_feature = []
	AU1_1_x = xlist[19:22]
	AU1_1_y = ylist[19:22]
	AU1_1_x,AU1_1_y = linear_interpolation(AU1_1_x,AU1_1_y)
	AU1_1_x,AU1_1_y = linear_interpolation(AU1_1_x,AU1_1_y)
	AU_feature = get_average_curvature(AU1_1_x,AU1_1_y)

	AU1_2_x = xlist[22:25]
	AU1_2_y = ylist[22:25]
	AU1_2_x,AU1_2_y = linear_interpolation(AU1_2_x,AU1_2_y)
	AU1_2_x,AU1_2_y = linear_interpolation(AU1_2_x,AU1_2_y)
	AU_feature = AU_feature + get_average_curvature(AU1_2_x,AU1_2_y)

	AU2_1_x = xlist[17:20]
	AU2_1_y = ylist[17:20]
	AU2_1_x,AU2_1_y = linear_interpolation(AU2_1_x,AU2_1_y)
	AU2_1_x,AU2_1_y = linear_interpolation(AU2_1_x,AU2_1_y)
	AU_feature = AU_feature + get_average_curvature(AU2_1_x,AU2_1_y)
	AU2_2_x = xlist[24:27]
	AU2_2_y = ylist[24:27]
	AU2_2_x,AU2_2_y = linear_interpolation(AU2_2_x,AU2_2_y)
	AU2_2_x,AU2_2_y = linear_interpolation(AU2_2_x,AU2_2_y)
	AU_feature = AU_feature + get_average_curvature(AU2_2_x,AU2_2_y)

	AU5_1_x = xlist[36:40]
	AU5_1_y = ylist[36:40]
	AU5_1_x,AU5_1_y = linear_interpolation(AU5_1_x,AU5_1_y)
	AU5_1_x,AU5_1_y = linear_interpolation(AU5_1_x,AU5_1_y)
	AU_feature = AU_feature + get_average_curvature(AU5_1_x,AU5_1_y)
	AU5_2_x = xlist[42:46]
	AU5_2_y = ylist[42:46]
	AU5_2_x,AU5_2_y = linear_interpolation(AU5_2_x,AU5_2_y)
	AU5_2_x,AU5_2_y = linear_interpolation(AU5_2_x,AU5_2_y)
	AU_feature = AU_feature + get_average_curvature(AU5_2_x,AU5_2_y)

	AU7_1_x = np.append(xlist[39:42],xlist[36])
	AU7_1_y = np.append(ylist[39:42],ylist[36])
	AU7_1_x,AU7_1_y = linear_interpolation(AU7_1_x,AU7_1_y)
	AU7_1_x,AU7_1_y = linear_interpolation(AU7_1_x,AU7_1_y)
	AU_feature = AU_feature + get_average_curvature(AU7_1_x,AU7_1_y)

	AU7_2_x = np.append(xlist[46:48],xlist[42])
	AU7_2_y = np.append(ylist[46:48],ylist[42])
	AU7_2_x,AU7_2_y = linear_interpolation(AU7_2_x,AU7_2_y)
	AU7_2_x,AU7_2_y = linear_interpolation(AU7_2_x,AU7_2_y)
	AU_feature = AU_feature + get_average_curvature(AU7_2_x,AU7_2_y)

	AU9_x = xlist[31:36]
	AU9_y = ylist[31:36]
	AU9_x,AU9_y = linear_interpolation(AU9_x,AU9_y)
	AU9_x,AU9_y = linear_interpolation(AU9_x,AU9_y)
	AU_feature = AU_feature + get_average_curvature(AU9_x,AU9_y)

	AU10_x = np.append(xlist[48:51],xlist[52:55])
	AU10_y = np.append(ylist[48:51],ylist[52:55])
	AU10_x,AU10_y = linear_interpolation(AU10_x,AU10_y)
	AU10_x,AU10_y = linear_interpolation(AU10_x,AU10_y)
	AU_feature = AU_feature + get_average_curvature(AU10_x,AU10_y)

	AU12_1_x = [xlist[48]] + [xlist[60]] + [xlist[67]]
	AU12_1_y = [ylist[48]] + [ylist[60]] + [ylist[67]]
	AU12_1_x,AU12_1_y = linear_interpolation(AU12_1_x,AU12_1_y)
	AU12_1_x,AU12_1_y = linear_interpolation(AU12_1_x,AU12_1_y)
	AU_feature = AU_feature + get_average_curvature(AU12_1_x,AU12_1_y)

	AU12_2_x = [xlist[54]] + [xlist[64]] + [xlist[65]]
	AU12_2_y = [ylist[54]] + [ylist[64]] + [ylist[65]]
	AU12_2_x,AU12_2_y = linear_interpolation(AU12_2_x,AU12_2_y)
	AU12_2_x,AU12_2_y = linear_interpolation(AU12_2_x,AU12_2_y)
	AU_feature = AU_feature + get_average_curvature(AU12_2_x,AU12_2_y)


	AU20_x = xlist[55:60]
	AU20_y = ylist[55:60]
	AU20_x,AU20_y = linear_interpolation(AU20_x,AU20_y)
	AU20_x,AU20_y = linear_interpolation(AU20_x,AU20_y)
	AU_feature = AU_feature + get_average_curvature(AU20_x,AU20_y)

	Norm_AU_feature = (AU_feature-np.min(AU_feature))/np.ptp(AU_feature)
	# print("AU feature", Norm_AU_feature)


	return Norm_AU_feature


def get_average_curvature(AU_xlist,AU_ylist):
	K = []
	Z = np.polyfit(AU_xlist,AU_ylist,3)
	P = np.poly1d(Z)
	P_1 = np.poly1d.deriv(P)
	P_2 = np.poly1d.deriv(P_1)
	for i in range(len(AU_xlist)):
		# K[i] =  P_2[AU_xlist[i]]/math.pow((1+math.pow((P_1(AU_xlist[i])),2)),1.5)
		Y = 1+math.pow(P_1(AU_xlist[i]),2)
		Y = math.pow(Y,1.5)
		# print("Y",Y)
		# print("X",P_2(AU_xlist[i]))
		K.append(P_2(AU_xlist[i])/Y)
	# m_K = np.mean(K)
	m_K = K
	return m_K


def get_vectorized_landmark(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)
	detections = detector(clahe_image, 1)
	for k,d in enumerate(detections): #For all detected face instances individually
		shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
		xlist = []
		ylist = []
		for i in range(0,68): #Store X and Y coordinates in two lists
			xlist.append(float(shape.part(i).x))
			ylist.append(float(shape.part(i).y))
		xmean = np.mean(xlist)
		ymean = np.mean(ylist)
		xcentral = [(x-xmean) for x in xlist]
		ycentral = [(y-ymean) for y in ylist]
		landmarks_dist = []
		landmarks_theta = []
		for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
			# landmarks_vectorized.append(w)
			# landmarks_vectorized.append(z)
			meannp = np.asarray((ymean,xmean))
			coornp = np.asarray((z,w))
			dist = np.linalg.norm(coornp-meannp)

			landmarks_dist.append(dist)
			landmarks_theta.append((math.atan2(y, x)*360)/(2*math.pi))

		landmarks_dist = landmarks_dist[17:]
		landmarks_theta = landmarks_theta[17:]
		landmarks_dist = np.array(landmarks_dist,dtype = np.float64)
		Norm_landmarks_dist = (landmarks_dist-np.min(landmarks_dist))/np.ptp(landmarks_dist)
		landmarks_theta = np.array(landmarks_theta,dtype = np.float64)
		Norm_landmarks_theta = (landmarks_theta-np.min(landmarks_theta))/np.ptp(landmarks_theta)

		landmarks_vectorized =  np.concatenate((Norm_landmarks_dist,Norm_landmarks_theta))

		# print("vectorized landmarks", landmarks_vectorized)
		return landmarks_vectorized
	if len(detections) < 1:
		landmarks_vectorized = np.array([])
	return landmarks_vectorized


def make_training_sets(w1,w2):
	training_data = np.array([])
	training_labels = np.array([])
	prediction_data = np.array([])
	prediction_labels = np.array([])
	for emotion in emotions:
		print(" working on %s" %emotion)
		training, prediction = get_datasets(emotion)
		#Append data to training and prediction list, and generate labels 0-7
		for item in training:
			image = cv2.imread(item) #open image
			# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
			# clahe_image = clahe.apply(gray)
			[xlist, ylist] = get_landmarks(image)
			vec_landmark = get_vectorized_landmark(image)*w1
			if (xlist.size) and (vec_landmark.size):
				Norm_AU_feature = extract_AU(xlist,ylist)*w2
				vec_AU = np.concatenate((Norm_AU_feature,vec_landmark))
				vec_AU = ((vec_AU-np.min(vec_AU))/np.ptp(vec_AU))

				training_labels = np.concatenate((training_labels,[emotions.index(emotion)]))
				if training_data.size:
					training_data = np.vstack((training_data,vec_AU))
					# training_data.append(data['landmarks_vectorised']) #append image array to training data list
				else:
					training_data = np.concatenate((training_data,vec_AU))
			else:
				print("no face detected on this training one")

				
		for item in prediction:
			image = cv2.imread(item)
			[xlist, ylist] = get_landmarks(image)
			vec_landmark = get_vectorized_landmark(image)*w1
			if (xlist.size) and (vec_landmark.size):
				Norm_AU_feature = extract_AU(xlist,ylist)*w2
				vec_AU = np.concatenate((Norm_AU_feature,vec_landmark))
				vec_AU = ((vec_AU-np.min(vec_AU))/np.ptp(vec_AU))
				prediction_labels = np.concatenate((prediction_labels,[emotions.index(emotion)]))
				if prediction_data.size:
					prediction_data = np.vstack((prediction_data,vec_AU))
					# training_data.append(data['landmarks_vectorised']) #append image array to training data list
				else:
					prediction_data = np.concatenate((prediction_data,vec_AU))
			else:
				print("no face detected on this prediction one")

	return training_data, training_labels, prediction_data, prediction_labels

def say(s):
	tts = gTTS(text=s, lang='en')
	tts.save('say.mp3')
	os.system("mpg321 say.mp3")


# Real-time
emotions = ["anger",  "disgust" ,"fear","happiness", "neutral", "sadness", "surprise"] #Emotion list
adjectives = ['angry', 'disgusted', 'afraid', 'happy', 'neutral', 'sad', 'surprised']

w1 = 0.75
w2 = 1-w1

cap = cv2.VideoCapture(0) #Webcam object
cap.set(3,320);
cap.set(4,240);
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
totalset=[0,0,0,0,0,0,0]

ports = pypot.dynamixel.get_available_ports()
print('available ports:', ports)  

if not ports:
	raise IOError('No port available.') 

port = ports[0]
print('Using the first on the list', port)

dxl_io = pypot.dynamixel.DxlIO(port)
print('Connected!')

found_ids = dxl_io.scan(range(13))
print('Found ids:', found_ids)

if len(found_ids) < 2:
	raise IOError('You should connect at least two motors on the bus for this test.')
#chose all motors and enable torque and set the same speed
ids = found_ids[:]
dxl_io.enable_torque(ids)
speed = dict(zip(ids, itertools.repeat(40)))
dxl_io.set_moving_speed(speed)

start_pose=[ -45, 0, 45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]
dxl_io.set_goal_position(dict(zip(ids, start_pose)))
time.sleep(1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

rectangleColor = (0,165,255)  

#cap = cv2.VideoCapture(0)
#cap.set(3,320);
#cap.set(4,240);

k=0.1	
xstep=.4*k
ystep=.4*k
target = 0
searchmode = 0
searchtime = time.time()
threadtime = time.time()
while True:
	realtime_data = np.array([])
	ret, frame = cap.read()
	# [xlist, ylist] = get_landmarks(frame)
	[xlist, ylist] = get_landmarks(frame)
	vec_landmark = get_vectorized_landmark(frame)*w1
	if (xlist.size) and (vec_landmark.size):
		Norm_AU_feature = extract_AU(xlist,ylist)*w2
		vec_AU = np.concatenate((Norm_AU_feature,vec_landmark))
		vec_AU = ((vec_AU-np.min(vec_AU))/np.ptp(vec_AU))
		realtime_data = np.concatenate((realtime_data,vec_AU))
		font = cv2.FONT_HERSHEY_SIMPLEX

	# print (realtime_data)

		clf = joblib.load('best_landmark_SVM.pkl') 
		Y = int(clf.predict([realtime_data]))
		print(type(Y))
		totalset[Y]=int(totalset[Y])+1
		
	##motor starts here

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	maxArea = 0  
	x = 0  
	y = 0  
	w = 0  
	h = 0

	for (_x,_y,_w,_h) in faces:
		if _w*_h > 0:
			x = _x
			y = _y
			w = _w
			h = _h
			maxArea = w*h
	if maxArea > 0:
		searchmode = 0
		cv2.rectangle(frame,(x,y),(x+w,y+h),rectangleColor,4)
		a = x+w/2
		b = y+h/2
		print ('face x coord',a)
		print ('face y coord',b)
		if 140<a<180 and 110<b<130:
			pass
		else:
			positions=dxl_io.get_present_position(ids)
			xpos=positions[0]
			ypos=positions[1]
			xnew=xpos-(a-160)*xstep
			ynew=ypos+(b-120)*ystep
			if -135<xnew<45 and -20<ynew<10:
				dxl_io.set_moving_speed(dict(zip([1], itertools.repeat(abs(a-160)/4))))
				dxl_io.set_moving_speed(dict(zip([2], itertools.repeat(abs(b-120)/4))))
				dxl_io.set_goal_position(dict(zip([1], [xnew])))
				dxl_io.set_goal_position(dict(zip([2], [ynew])))
				if xnew < xpos:
					target = 0
				else:
					target = 1

			# else:
			# 	dxl_io.set_goal_position(dict(zip([1], [xpos])))
			# 	dxl_io.set_goal_position(dict(zip([2], [ypos])))
			print('new',xnew, ynew)

			time.sleep(.01)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		# cv2.putText(frame, "x: {}, y: {}".format(x+(1/2)*w, y+(1/2)*h), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		# 1, (0, 0, 255), 5)
		#cv2.putText(frame, "x: {}, y: {}".format(a, b), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		#1, (0, 0, 255), 5)
		cv2.putText(frame,emotions[Y],(50,70), font, 2,(0,0,255),2)
		# eyes = eye_cascade.detectMultiScale(roi_gray)
		# for (ex,ey,ew,eh) in eyes:
		#     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		searchtime = time.time()
		if random.random() < 0.1 and time.time()-threadtime>10.5:
			print('randomd')
			if random.randint(1,2)==1:
				print('wavin')
				Thread(target=wave).start()
			else:
				print('lwavin')
				Thread(target=waveleft).start()
			threadtime=time.time()	
			time.sleep(5)
	elif time.time()-searchtime > .5:
		if target == 0:
			if searchmode == 0:
				dxl_io.set_moving_speed(dict(zip([1,2], itertools.repeat(20))))
				dxl_io.set_goal_position(dict(zip([2], [-10]))) # only looking side to side right now. might change later
				dxl_io.set_goal_position(dict(zip([1], [-95])))
				searchmode = 1
			if dxl_io.get_present_position([1])[0] <= -90:
				target = 1
				searchmode = 0
		else:
			if searchmode == 0:
				dxl_io.set_moving_speed(dict(zip([1,2], itertools.repeat(20))))
				dxl_io.set_goal_position(dict(zip([2], [-10]))) # only looking side to side right now. might change later
				dxl_io.set_goal_position(dict(zip([1], [5])))
				searchmode = 1
			if dxl_io.get_present_position([1])[0] >= 0:
				target = 0
				searchmode = 0

	# while maxArea == 0:
	# 	dxl_io.set_moving_speed(dict(zip(ids, itertools.repeat(40))))
	# 	dxl_io.set_goal_position(dict(zip([2], [0]))) # only looking side to side right now. might change later
	# 	dxl_io.set_goal_position(dict(zip([1], [-135])))
	# 	while dxl_io.get_present_position([1])[0] > -130:
	# 		for (_x,_y,_w,_h) in faces:
	# 			if _w*_h > maxArea:
	# 				x = _x
	# 				y = _y
	# 				w = _w
	# 				h = _h
	# 				maxArea = w*h
	# 		if maxArea > 0:
	# 			break
	# 	if maxArea>0:
	# 		break
	# 	dxl_io.set_goal_position(dict(zip([1], [45])))
	# 	while dxl_io.get_present_position([1])[0] <40:
	# 		for (_x,_y,_w,_h) in faces:
	# 			if _w*_h > maxArea:
	# 				x = _x
	# 				y = _y
	# 				w = _w
	# 				h = _h
	# 				maxArea = w*h
	# 		if maxArea > 0:
	# 			break
	# 	if maxArea>0:
	# 		break	


	frame = cv2.resize(frame,(320*4, 240*4))
	cv2.imshow("image", frame) #Display the frame

	if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
		cv2.destroyAllWindows()
		break

"""RESULTS OMITTED FOR SHOWCASE DEMO
print(totalset)
print(max(totalset))
for i in range(len(totalset)):
	print(i)
	print(totalset[i])
	if totalset[i] == max(totalset):
		print(emotions[i])
		say(('Your emotion is {}').format(adjectives[i]))
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
patches, texts = plt.pie(totalset, colors=colors, shadow=True, startangle=90)
plt.legend(patches, emotions, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
"""

## Training
# emotions = ["anger",  "disgust" ,"fear","happiness", "neutral", "sadness", "surprise"] #Emotion list

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# training_data = np.array([])
# training_labels = np.array([])
# prediction_data = np.array([])
# prediction_labels = np.array([])

# # CC = [0.1,1,10,50,100,500,1000,3000, 10000, 30000]
# # for i in range (10):
# #     C = CC[i]
# #     print("C right now is:", C)
# #     for j in range (18):
# #         w1 = 0.05*(j+1)
# #         w2 = 1-w1
# #         print("right now w1 is:", w1)
# C = 100
# w1 = 0.75
# w2 = 1-w1
# # w1=1
# # w2=1
# start = time.time()
# # np.savetxt('training_data.txt', training_data, fmt='%1.4e')
# # np.savetxt('training_labels.txt', training_labels, fmt='%1.4e')
# # np.savetxt('prediction_data.txt', prediction_data, fmt='%1.4e')
# # np.savetxt('prediction_labels.txt', prediction_labels, fmt='%1.4e')
# clf = SVC(C = C, kernel='linear', probability=True, tol=1e-3)
# accuracy_inside = []
# accur_lin = []

# for m in range(0,1):
#     # print("Making sets %s" %m) #Make sets by random sampling 80/20%
#     [training_data, training_labels, prediction_data, prediction_labels] = make_training_sets(w1,w2)
#     # print("training SVM linear %s" %m) #train SVM
#     clf.fit(training_data, training_labels)
#     print("getting accuracies %s" %m) #Use score() function to get accuracy
#     pred_lin = clf.score(prediction_data, prediction_labels)
#     print ("linear: ", pred_lin)
#     accuracy_inside.append(pred_lin) #Store accuracy in a list
#     joblib.dump(clf, 'best_landmark_SVM.pkl')
# accur_lin.append(accuracy_inside)
# end = time.time()
# elapsed = end - start
# print("time",elapsed)
