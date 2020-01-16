import itertools
import numpy as np
import time
import serial
import pypot.dynamixel
import time
#import RPi.GPIO as GPIO

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
speed = dict(zip(ids, itertools.repeat(20)))
dxl_io.set_moving_speed(speed)
dxl_io.set_moving_speed(dict(zip([1,2], itertools.repeat(55))))
dxl_io.set_moving_speed(dict(zip([3,8], itertools.repeat(25))))

start_pose=[ -45, -9.53, 45.6, 30.06, -7.5, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]

def speed_traj(ID, stepsize, begin_pose, end_pose, v_max):
    speed = 0
    pose = begin_pose
    t = round(2*abs(end_pose-begin_pose)/v_max,2)
    t1 = round(t/2,2)
    t2 = t
    print('t2',t)
    a = np.arange(0,t1,stepsize)
    for t in range (a.shape[0]):
        speed = v_max/(t1/stepsize)+speed
        print('speed1',speed)
        pose = pose+ (abs(end_pose-begin_pose)/(end_pose-begin_pose))*speed*stepsize
        time.sleep(stepsize)
        dxl_io.set_moving_speed(dict(zip([ID], itertools.repeat(speed))))
        dxl_io.set_goal_position(dict(zip([ID], [pose])))
        print('pose',pose)
    for t in range (a.shape[0]):
        speed = speed - v_max/((t2-t1)/stepsize)
        print('speed2',speed)
        pose = pose+ (abs(end_pose-begin_pose)/(end_pose-begin_pose))*speed*stepsize
        time.sleep(stepsize)
        dxl_io.set_moving_speed(dict(zip([ID], itertools.repeat(speed))))
        dxl_io.set_goal_position(dict(zip([ID], [pose])))

def test():
    dxl_io.set_goal_position(dict(zip(ids, start_pose)))
    time.sleep(1.5)
    speed_traj(1, 0.25, -55.28, -15, 45)
    time.sleep(0.5)
    speed_traj(1, 0.25, -15, -90, 45)
    print("finish test")

#define functions of different behavior

def shake_head(E):
    dxl_io.set_goal_position(dict(zip(ids, start_pose)))
    dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(55))))
    time.sleep(1.5)
    dxl_io.set_goal_position(dict(zip([1], [-15+10*E])))
    time.sleep(2)
    dxl_io.set_goal_position(dict(zip([1], [-75-10*E])))
    time.sleep(2)
    
def nod_head(E):
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

def wave(E):
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

def handshake(E):
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


while True:
	extraversion = 0
	agreeableness = 0
	print(type(extraversion))
	while extraversion<1 or extraversion>5:
		extraversion = int(input('Please input an extraversion score between 1 and 5\n'))
		print(type(extraversion))
	while agreeableness<1 or agreeableness>5:
		agreeableness = int(input('Please input an agreeableness score between 1 and 5\n'))
	pausetime = (6-agreeableness)/2
	wave(extraversion)
	print('hello')
	time.sleep(pausetime)
	shake_head(extraversion)
	print('no')
	time.sleep(pausetime)
	handshake(extraversion)
	print('nice to meet you')
	time.sleep(pausetime)
	nod_head(extraversion)
	print('yes')
	time.sleep(pausetime)

pwm1.stop()
pwm2.stop()
GPIO.cleanup()
