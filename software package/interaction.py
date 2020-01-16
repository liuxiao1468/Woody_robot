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
import speech_recognition as sr #replace???

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

#speech recognition function
def recognize_speech_from_mic(recognizer, microphone, show):
	"""Transcribe speech from recorded from `microphone`.

	Returns a dictionary with three keys:
	"success": a boolean indicating whether or not the API request was
			   successful
	"error":   `None` if no error occured, otherwise a string containing
			   an error message if the API could not be reached or
			   speech was unrecognizable
	"transcription": `None` if speech could not be transcribed,
			   otherwise a string containing the transcribed text
	"""
	# check that recognizer and microphone arguments are appropriate type
	if not isinstance(recognizer, sr.Recognizer):
		raise TypeError("`recognizer` must be `Recognizer` instance")

	if not isinstance(microphone, sr.Microphone):
		raise TypeError("`microphone` must be `Microphone` instance")

	# adjust the recognizer sensitivity to ambient noise and record audio
	# from the microphone
	with microphone as source:
		#recognizer.adjust_for_ambient_noise(source, duration=0.1)
		print('Listening')
		audio = recognizer.listen(source,phrase_time_limit=3)
		print('Recognizing')

	# set up the response object
	response = {
		"success": True,
		"error": None,
		"transcription": None
	}

	# try recognizing the speech in the recording
	# if a RequestError or UnknownValueError exception is caught,
	#     update the response object accordingly
	try:
		response["transcription"] = recognizer.recognize_google(audio, show_all=show)
	except sr.RequestError:
		# API was unreachable or unresponsive
		response["success"] = False
		response["error"] = "API unavailable"
	except sr.UnknownValueError:
		# speech was unintelligible
		response["error"] = "Unable to recognize speech"

	return response


#define functions of different behavior
def perform(E, emotion):
	g = random.randint(1,7)
	print(g)
	if g == 1:
		print('Shaking head')
		shake_head(E, emotion)
	elif g == 2:
		print('Nodding head')
		nod_head(E,emotion)
	elif g == 3:
		print('Hugging')
		hug(E,emotion)
	elif g == 4:
		print('Thanking')
		thank_you(E,emotion)
	elif g == 5:
		print('Dancing 1')
		dance1(E,emotion)
	elif g == 6:
		print('Dancing 2')
		dance2(E,emotion)
	

def shake_head(E, emotion):
	sayings = [ #UPDATE NEEDED!!!
	'Don\'t be mad. Have a hug and calm down.',
	'disgust',
	'I will hug you to keep you safe.',
	'I am happy too. Come give me a hug',
	''
	'I can hug you to cheer you up',
	'Surprise! It\'s hugging time.']
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(55))))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([1], [-15+10*E])))
	time.sleep(2)
	dxl_io.set_goal_position(dict(zip([1], [-75-10*E])))
	time.sleep(2)
	
def nod_head(E, emotion):
	sayings = [ #UPDATE NEEDED!!!
	'Don\'t be mad. Have a hug and calm down.',
	'disgust',
	'I will hug you to keep you safe.',
	'I am happy too. Come give me a hug',
	''
	'I can hug you to cheer you up',
	'Surprise! It\'s hugging time.']
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(55))))
	time.sleep(1.5)
	nod=10+2*E #previously 16.25
	dxl_io.set_goal_position(dict(zip([2], [nod])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([2], [-9.53])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([2], [nod])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([2], [-9.53])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([2], [nod])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([2], [-9.53])))
	time.sleep(1)

def wave(E, emotion):
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(55)))) #this still needs to change
	dxl_io.set_goal_position(dict(zip([3], [-70])))
	time.sleep(3)
	espeed=35+10*E
	dxl_io.set_moving_speed(dict(zip([5], itertools.repeat(65))))
	extend=-7.5-6*E #formerly -36.2
	dxl_io.set_goal_position(dict(zip([5], [extend])))
	time.sleep(0.5)#this might have to change with new speed, or wait for position reached/motion stopped
	dxl_io.set_goal_position(dict(zip([5], [-7.5])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [extend])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-7.5])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [extend])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-7.5])))
	time.sleep(0.5)
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3], [32.7])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([3], [40])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

def handshake(E, emotion):
	sayings = [ #UPDATE NEEDED!!!
	'Don\'t be mad. Have a hug and calm down.',
	'disgust',
	'I will hug you to keep you safe.',
	'I am happy too. Come give me a hug',
	''
	'I can hug you to cheer you up',
	'Surprise! It\'s hugging time.']
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([8], itertools.repeat(60))))
	dxl_io.set_goal_position(dict(zip([8], [15.69+E])))
	time.sleep(2.5)
	dxl_io.set_moving_speed(dict(zip([10], itertools.repeat(55))))
	dxl_io.set_goal_position(dict(zip([10], [62.32])))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([12], itertools.repeat(60))))
	dxl_io.set_goal_position(dict(zip([12], [0])))
	time.sleep(0.5)
	dxl_io.set_moving_speed(dict(zip([8], itertools.repeat(40))))
	dxl_io.set_goal_position(dict(zip([8], [-5-E])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([8], [15.69+E])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([8], [-5-E])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([8], [15.69+E])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip([12], [50])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [8.94])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([8], [-35])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)

def hug(E, emotion):
	sayings = [
	'Don\'t be mad. Have a hug and calm down.',
	'disgust',
	'I will hug you to keep you safe.',
	'I am happy too. Come give me a hug',
	''
	'I can hug you to cheer you up',
	'Surprise! It\'s hugging time.']
	say(sayings[emotion])
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)
	dxl_io.set_goal_position(dict(zip([2], [16.25])))
	time.sleep(2)
	dxl_io.set_goal_position(dict(zip([2], [-9.53])))
	time.sleep(1)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(60))))
	dxl_io.set_goal_position(dict(zip([3, 8], [-10, 10])))
	time.sleep(3)
	dxl_io.set_moving_speed(dict(zip([4,9], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([4, 9], [58.21, -62.02])))
	time.sleep(3)
	dxl_io.set_moving_speed(dict(zip([7,12], itertools.repeat(60))))
	dxl_io.set_goal_position(dict(zip([7, 12], [-21.85, -24.78])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7, 12], [-46.77, 50.0])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7, 12], [-21.85, -24.78])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7, 12], [-46.77, 50.0])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([4, 9], [30.06, -27.13])))
	time.sleep(2)
	dxl_io.set_goal_position(dict(zip([4, 9], [58.21, -62.02])))
	time.sleep(3)
	dxl_io.set_goal_position(dict(zip([4, 9], [30.06, -27.13])))
	time.sleep(2)
	dxl_io.set_moving_speed(dict(zip([3,8], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3,8], [32.7, -32.11])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

def thank_you(E, emotion):
	sayings = [ #UPDATE NEEDED!!!
	'Don\'t be mad. Have a hug and calm down.',
	'disgust',
	'I will hug you to keep you safe.',
	'I am happy too. Come give me a hug',
	''
	'I can hug you to cheer you up',
	'Surprise! It\'s hugging time.']
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(55))))
	dxl_io.set_goal_position(dict(zip([3, 8], [23.8, -24.19])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([2, 3, 8], [-9.53, -2.79, 2.49])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([2], [16.25])))
	time.sleep(0.5)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3, 8], [32.7, -32.11])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

def dance1(E,emotion):
	sayings = [ #UPDATE NEEDED!!!
	'Don\'t be mad. Have a hug and calm down.',
	'disgust',
	'I will hug you to keep you safe.',
	'I am happy too. Come give me a hug',
	''
	'I can hug you to cheer you up',
	'Surprise! It\'s hugging time.']
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(55))))
	dxl_io.set_goal_position(dict(zip([3, 8], [1.32, -6.83])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5, 10], [-64.06, 63.78])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([3, 8], [-86.07, 87.54])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([1, 5, 10], [-81.96, -38.27, 39.85])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([1, 5, 10], [-6.3, -89.88, 43.84])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([1, 5, 10], [-81.96, -38.27, 39.85])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([1, 5, 10], [-6.3, -89.88, 43.84])))
	time.sleep(0.5)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3, 8], [32.7, -32.11])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

def dance2(E,emotion):
	sayings = [ #UPDATE NEEDED!!!jones
	'Don\'t be mad. Have a hug and calm down.',
	'disgust',
	'I will hug you to keep you safe.',
	'I am happy too. Come give me a hug',
	''
	'I can hug you to cheer you up',
	'Surprise! It\'s hugging time.']
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(55))))
	dxl_io.set_goal_position(dict(zip([3, 8], [4.84, -5.43])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5, 10, 7, 12], [-39.15, 45.01, -12.40, 8.05])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [16.25])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-39.15])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [-59.09])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [45.01])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [16.25])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([5], [-39.15])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [-59.09])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([10], [45.01])))
	time.sleep(1)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3, 8], [32.7, -32.11])))
	time.sleep(1)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

#main user data function
def userdata(user):
	found = 0
	f=open("data.txt", "r")
	fl=f.readlines()
	for i in range(int((len(fl)/6))):
		j=6*i
		if fl[j]=='{}\n'.format(user):
			found = 1
			e = fl[j+1]
			a = fl[j+2]
			c = fl[j+3]
			n = fl[j+4]
			o = fl[j+5]
			f.close()
			say(('Hi {}, welcome back.').format(user))
			break
	if found == 0:
		say(('Nice to meet you, {}. I\'d like to ask you some questions to get to know you').format(user))
		# set the list of options, questions, and initialize results vector
		OPTIONS = ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree']
		#new things to search for. need to figure out how to correctly search the data from the speech recognition to proceed
		agr = ['agree', 'yes', 'support', 'affirm', 'yeah', 'agreement', 'affirmative', 'of course']
		dis = ['disagree', 'no', 'differ', 'oppose', 'disagreement', 'negative', 'never', 'of course not']
		neu = ['neutral', 'maybe', 'neither', 'both', 'sometimes'] #d 'i don't know', detect full words
		mod = ['strongly', 'a lot', 'greatly', 'very much', 'really', 'strong', 'absolutely']
		neg = ['don\'t', 'do not']
		QUESTIONS = [
		'is reserved',
		'is generally trusting',
		'tends to be lazy',
		'is relaxed, handles stress well',
		'has few artistic interests',
		'is outgoing, sociable',
		'tends to find fault with others',
		'does a thorough job',
		'gets nervous easily',
		'has an active imagination'
		]
		r = []

		# create recognizer and mic instances
		recognizer = sr.Recognizer()
		microphone = sr.Microphone()

		# format the instructions string DO THIS. REITERATE, DICTIONARY, START GESTURES
		instructions = ('\nPlease answer the following questions with your level of agreement.\n')
		
		# show instructions and wait 1 second before starting the test
		print(instructions)
		#say(instructions)
		#time.sleep(1)

		for i in range(10): 
			question = (
				"Question {number}: Do you see yourself as someone who {prompt}?\n"
			).format(number=(i+1),prompt=QUESTIONS[i])
			print(question)
			say(question)
			while True:
				answer = recognize_speech_from_mic(recognizer, microphone, True)
				time.sleep(1)   
				match=False
				if not answer["success"]:
					break
				if answer['transcription']: 
					words=answer['transcription'].get('alternative')
					for j in range(len(words)):
						g=words[j].get('transcript')
						for k in range(len(dis)):
							print(g)
							print(dis[k],'\n')
							if dis[k] in g:
								result=2
								match=True
								phrase=g
								break
						if match==True:
							break
						for k in range(len(neu)):
							print(g)
							print(neu[k],'\n')
							if neu[k] in g:
								result=3
								match=True
								phrase=g
								break
						if match==True:
							break
						for k in range(len(agr)):
							print(g)
							print(agr[k],'\n')
							if agr[k] in g:
								result=4
								match=True
								phrase=g
								break
						if match==True:
							break
					if match == True:
						for j in range(len(mod)):
							print(phrase)
							print(mod[j],'\n')
							if mod[j] in phrase:
								result=2*result-3
								break
						for j in range(len(neg)):
							print(phrase)
							print(neg[j],'\n')
							if neg[j] in phrase:
								result=6-result
								break
						print(result)
						r.append(result)
						break
					print(('Invalid Response: {}\n').format(answer['transcription']))
					say('Invalid Response.')
					print(type(answer['transcription']))
				else:
					print("I didn't catch that. What did you say?\n")
					say("I didn't catch that. What did you say?\n")
		
			# if there was an error, stop the test
			if answer["error"]:
				print("ERROR: {}".format(answer["error"]))
				break

			# show the user the transcription
			print("You said: {}\n".format(answer["transcription"]))

		#Calculate final score
		e = (-r[0]+r[5]+6)/2
		a = (+r[1]-r[6]+6)/2
		c = (-r[2]+r[7]+6)/2
		n = (-r[3]+r[8]+6)/2
		o = (-r[4]+r[9]+6)/2
		f=open('data.txt','a')
		f.write(('{user}\n{e}\n{a}\n{c}\n{n}\n{o}\n').format(user=user,e=e,a=a,c=c,n=n,o=o))
		f.close()
	result = [float(e), float(a), float(c), float(n), float(o)]
	return result

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

start_pose=[ -45, 0]
dxl_io.set_goal_position(dict(zip(ids, start_pose)))

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
initloop = 1
gesture = 0
agreeableness=0
Y = 4


#main loop. Still needs gestures.
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
		totalset[Y]=int(totalset[Y])+1
		cv2.putText(frame,emotions[Y],(50,70), font, 2,(255,255,255),2)

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
			time.sleep(.01)

		gesturechance = random.random()
		if gesturechance < agreeableness/20:
			perform(extraversion, Y)

		if initloop == 1:
			wave(3,1)
			say('Hello! What is your name?')
			name = input('Please input name here\n')
			personality = userdata(name)
			extraversion = personality[0]
			print(extraversion)
			print(type(extraversion	))
			agreeableness = personality[1]
			consciensciousness = personality[2]
			neuroticism = personality[3]
			openness = personality[4]
			initloop = 0


		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		cv2.putText(frame, "x: {}, y: {}".format(a, b), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		1, (0, 0, 255), 5)
		searchtime = time.time()
	elif time.time()-searchtime > .5:
		if target == 0:
			if searchmode == 0:
				dxl_io.set_moving_speed(dict(zip(ids, itertools.repeat(20))))
				dxl_io.set_goal_position(dict(zip([2], [-10]))) # only looking side to side right now. might change later
				dxl_io.set_goal_position(dict(zip([1], [-135])))
				searchmode = 1
			if dxl_io.get_present_position([1])[0] <= -130:
				target = 1
				searchmode = 0
		else:
			if searchmode == 0:
				dxl_io.set_moving_speed(dict(zip(ids, itertools.repeat(20))))
				dxl_io.set_goal_position(dict(zip([2], [-10]))) # only looking side to side right now. might change later
				dxl_io.set_goal_position(dict(zip([1], [45])))
				searchmode = 1
			if dxl_io.get_present_position([1])[0] >= 40:
				target = 0
				searchmode = 0
		if time.time()-searchtime > 20:
			initloop = 1
			#save results if any found

	cv2.imshow("image", frame) #Display the frame

	if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
		cv2.destroyAllWindows()
		break
print(totalset)
print(max(totalset))
for i in range(len(totalset)):
	print(i)
	print(totalset[i])
	if totalset[i] == max(totalset):
		print(emotions[i])
		say(('You seem {} today.').format(adjectives[i]))
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
patches, texts = plt.pie(totalset, colors=colors, shadow=True, startangle=90)
plt.legend(patches, emotions, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
