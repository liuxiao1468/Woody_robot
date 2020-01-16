import time
import speech_recognition as sr
from gtts import gTTS
import os

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

#text to speech function
def say(s):
	tts = gTTS(text=s, lang='en')
	tts.save('say.mp3')
	os.system("mpg321 say.mp3")



if __name__ == "__main__":
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

		# get the response from the user
		# if a transcription matches the options, add to response list
		# if the length of the response increased, break out of the loop
		#     and continue
		# if no transcription returned and API request failed, break
		#     loop and continue
		# if API request succeeded but no transcription was returned,
		#     re-prompt the user to say their guess again. Do this up
		#     to PROMPT_LIMIT times
		while True:
			answer = recognize_speech_from_mic(recognizer, microphone)
			time.sleep(1)   
			match=False           
			#for j in range(5):
				#if ('\'transcript\': \'{}\''.format(OPTIONS[j])) in str(answer['transcription']):
				#if answer['transcription'] == OPTIONS[j]:
					#r.append(j+1)
					#match=True
					#break
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
	E = (-r[0]+r[5]+6)/2
	A = (+r[1]-r[6]+6)/2
	C = (-r[2]+r[7]+6)/2
	N = (-r[3]+r[8]+6)/2
	O = (-r[4]+r[9]+6)/2
	s = ('\nScores:\n')
	e = ('Extraversion: {}\n').format(E)
	a = ('Agreeableness: {}\n').format(A)
	c = ('Conscientiousness: {}\n').format(C)
	n = ('Neuroticism: {}\n').format(N)
	o = ('Openness: {}\n').format(O)
	scores=[s,e,a,c,n,o]
	print(('{s}{e}{a}{c}{n}{o}').format(s=s,e=e,a=a,c=c,n=n,o=o))
	say(('{s},{e},{a},{c},{n},{o}\n').format(s=s,e=e,a=a,c=c,n=n,o=o))
