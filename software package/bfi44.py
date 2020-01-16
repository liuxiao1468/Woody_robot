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
        #recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source,phrase_time_limit=2)

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
        response["transcription"] = recognizer.recognize_google(audio)
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
    os.system("mpg123 say.mp3")



if __name__ == "__main__":
    # set the list of options, questions, and initialize results vector
    OPTIONS = ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree']
    QUESTIONS = [
    'is talkative', 
    'tends to find fault with others',
    'does a thorough job',
    'is depressed, blue',
    'is original, comes up with new ideas',
    'is reserved',
    'is helpful and unselfish with others',
    'can be somewhat careless',
    'is relaxed, handles stress well',
    'is curious about many different things',
    'is full of energy',
    'starts quarrels with others',
    'is a reliable worker',
    'can be tense',
    'is ingenious, a deep thinker',
    'generates a lot of enthusiasm',
    'has a forgiving nature',
    'tends to be disorganized',
    'worries a lot',
    'has an active imagination',
    'tends to be quiet',
    'is generally trusting',
    'tends to be lazy',
    'is emotionally stable, not easily upset',
    'is inventive',
    'has an assertive personality',
    'can be cold and aloof',
    'perseveres until the task is finished',
    'can be moody',
    'values artistic, aesthetic experiences',
    'is sometimes shy, inhibite',
    'is considerate and kind to almost everyone',
    'does things efficiently',
    'remains calm in tense situation',
    'prefers work that is routine',
    'is outgoing, sociable',
    'is sometimes rude to others',
    'makes plans and follows through with them',
    'gets nervous easily',
    'likes to reflect, play with ideas',
    'has few artistic interests',
    'likes to cooperate with others',
    'is easily distracted',
    'is sophisticated in art, music, or literature'
    ]
    r = []

    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # format the instructions string
    instructions = ('\nPlease answer the following questions with your level of agreement.\n')
    
    # show instructions and wait 1 second before starting the test
    print(instructions)
    say(instructions)
    time.sleep(1)

    for i in range(44): 
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
            for j in range(5):
                #if answer['transcription'] == str(j+1):
                if answer['transcription'] == OPTIONS[j]:
                    r.append(j+1)
                    break
            if len(r) == i+1:
                break 
            if not answer["success"]:
                break
            if answer['transcription']: 
                wrong=('You said {}. Acceptable responses are strongly agree, agree, neutral, disagree, and strongly disagree\n').format(answer['transcription'])
                print(wrong)
                say(wrong)
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
    E = (+r[0]-r[5]+r[10]+r[15]-r[20]+r[25]-r[30]+r[35]+18)/8
    A = (-r[1]+r[6]-r[11]+r[16]+r[21]-r[26]+r[31]-r[36]+r[41]+24)/9
    C = (+r[2]-r[7]+r[12]-r[17]-r[22]+r[27]+r[32]+r[37]-r[42]+24)/9
    N = (+r[3]-r[8]+r[13]+r[18]-r[23]+r[28]-r[33]+r[38]+18)/8
    O = (+r[4]+r[9]+r[14]+r[19]+r[24]+r[29]-r[34]+r[39]-r[40]+r[43]+12)/10
    s = ('\nScores:\n')
    e = ('Extraversion: {}\n').format(E)
    a = ('Agreeableness: {}\n').format(A)
    c = ('Conscientiousness: {}\n').format(C)
    n = ('Neuroticism: {}\n').format(N)
    o = ('Openness: {}\n').format(O)
    scores=[s,e,a,c,n,o]
    print(('{s}{e}{a}{c}{n}{o}').format(s=s,e=e,a=a,c=c,n=n,o=o))
    for i in scores:
        say(i)
