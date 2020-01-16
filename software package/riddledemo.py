import time
import speech_recognition as sr
from gtts import gTTS
import os
import itertools
import pypot.dynamixel
import threading
import numpy as np
import cv2
import socket



#speech recognition function
def recognize_speech_from_mic(recognizer, microphone):
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
		response["transcription"] = recognizer.recognize_google(audio, show_all=True)
	except sr.RequestError:
		# API was unreachable or unresponsive
		response["success"] = False
		response["error"] = "API unavailable"
	except sr.UnknownValueError:
		# speech was unintelligible
		response["error"] = "Unable to recognize speech"

	return response

def wave():
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
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
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

def thinking():
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(55)))) #this still needs to change
	dxl_io.set_goal_position(dict(zip([3], [-90])))
	time.sleep(3)
	dxl_io.set_moving_speed(dict(zip([7], itertools.repeat(65))))
	dxl_io.set_goal_position(dict(zip([7], [0])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7], [-44.13])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7], [0])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7], [-44.13])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7], [0])))
	time.sleep(0.5)
	dxl_io.set_goal_position(dict(zip([7], [-44.13])))
	time.sleep(0.5)
	dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3], [32.7])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([3], [40])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

def correct():
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3, 5, 8, 10], itertools.repeat(55))))
	dxl_io.set_goal_position(dict(zip([3, 8], [0, 0])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([5, 10], [-90, 90])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip([3, 8], [-90, 90])))
	time.sleep(1.5)
	dxl_io.set_moving_speed(dict(zip([3, 8], itertools.repeat(30))))
	dxl_io.set_goal_position(dict(zip([3, 8], [32.7, -32.11])))
	time.sleep(1.5)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)

def yesorno(recognizer, microphone):  
	match=False     
	while True:
		answer = recognize_speech_from_mic(recognizer, microphone)
		time.sleep(1) 
		if answer['transcription']: 
			words=answer['transcription'].get('alternative')
			for j in range(len(words)):
				g=words[j].get('transcript')
				if 'no' in g:
					yesresponse = 0
					match = True
				elif 'yes' in g:
					yesresponse = 1
					match = True
				if match==True:
					break
			if match == True:
				break
			print(('Invalid Response: {}\n').format(answer['transcription']))
			say('Invalid Response.')
			print(type(answer['transcription']))
		else:
			print("I didn't catch that. What did you say?\n")
			say("I didn't catch that. What did you say?\n")
	print(yesresponse)
	return yesresponse

def getanswer(recognizer, microphone):
	match=False     
	while True:
		answer = recognize_speech_from_mic(recognizer, microphone)
		time.sleep(1) 
		if answer['transcription']: 
			words=answer['transcription'].get('alternative')
			for j in range(len(words)):
				g=words[j].get('transcript')
				if 'Woody' in g:
					correct = 1
					match = True
				else:
					correct = 0
					match = True
				if match==True:
					break
			if match == True:
				break
			#print(('Invalid Response: {}\n').format(answer['transcription']))
			say('I\'m not sure that\'s correct.')
			#print(type(answer['transcription']))
		else:
			print("I didn't catch that. What did you say?\n")
			say("I didn't catch that. What did you say?\n")
	print(correct)
	return correct

def say(s):
	tts = gTTS(text=s, lang='en')
	tts.save('say.mp3')
	os.system("mpg321 say.mp3")

def hello():
	say('Hello, my name is Woody. Nice to meet you!')

def askriddle():
	say('Would you like me to tell you a riddle?')

def tellriddle():
	say('This one is a head scratcher. What is awesome and rhymes with hoodie?')


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
speed = dict(zip(ids, itertools.repeat(30)))
dxl_io.set_moving_speed(speed)

start_pose=[ -45, 0, 45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]
dxl_io.set_goal_position(dict(zip(ids, start_pose)))
target = 0
searchmode = 0

if __name__=='__main__':
	recognizer = sr.Recognizer()
	microphone = sr.Microphone()


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
		thread_wave = threading.Thread(target = wave)
		thread_hello = threading.Thread(target = hello)
		thread_tellriddle = threading.Thread(target = tellriddle)
		thread_thinking = threading.Thread(target = thinking)
		thread_correct = threading.Thread(target = correct)
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
			break
		else:
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


		thread_wave.start()
		time.sleep(2)
		thread_hello.start()

		
		time.sleep(10)
		askriddle()
		time.sleep(2)
		response = yesorno(recognizer, microphone)
		attempts = 0
		if response == 1:
			thread_tellriddle.start()
			thinking()
			while True:
				answer = getanswer(recognizer, microphone)
				if answer == 1:
					thread_correct.start()
					say('I think you\'re right! Well done!')
					time.sleep(10)
					break
				elif attempts == 0:
					say('Why don\'t you try again?')
					attempts = 1
					time.sleep(10)
				else:
					thread_correct.start()
					say('Wait, I solved it myself! The answer is Woody!')
					time.sleep(10)
					break
		else:
			say('All right. How are you doing today?')
			time.sleep(3)
			#recognize_speech_from_mic(recognizer, microphone)
		say('It was my pleasure meeting you. Please stop by again! Bye!')