import time
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BOARD)
servo1=37
#servo2=13
GPIO.setup(servo1,GPIO.OUT)
#GPIO.setup(servo2,GPIO.OUT)
pwm1=GPIO.PWM(servo1,50)
pwm1.start(7)
#time.sleep(2)
#pwm2=GPIO.PWM(servo2,50)
#pwm2.start(7)
for i in range(0,20):
        E=chr(69)
	desiredPosition=input("Where do you want the servo(-90-90), press E to skip:")
	if (desiredPosition == E):
                break
        else:
                DC=1./18.*(desiredPosition+90)+1
                #DC=12
                pwm1.ChangeDutyCycle(DC)
                #pwm2.ChangeDutyCycle(DC)
                time.sleep(1)
pwm1.stop()
#pwm2.stop()
GPIO.cleanup()
