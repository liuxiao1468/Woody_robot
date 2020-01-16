import itertools
import numpy as np
import time
import serial
import pypot.dynamixel
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

# while True:
# 	positions=dxl_io.get_present_position([1, 2])
# 	speeds = dxl_io.get_present_speed([1,2])
# 	print(positions[0])
# 	print(positions[1])
# 	print(speeds[0])
# 	print(speeds[1])
# 	time.sleep(1)

#dxl_io.enable_torque(ids)
setattr(self, motor_name, 'jim')
setattr(self, motor_name, 'stan')
woody = pypot.robot.Robot(motor_controllers=[], sensor_controllers=[], sync = True)
woody.goto_position(position_for_motors=dict(zip(['jim', 'stan'], [0,0])), duration=1, control = None, wait = False)
time.sleep(3)
dxl_io.disable_torque(ids)