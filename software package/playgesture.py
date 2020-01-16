import itertools
import time
import pypot.dynamixel
from universalgesture import perform

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
start_pose=[ -45, -9.53, 45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]

dxl_io.enable_torque(ids)
speed = dict(zip(ids, itertools.repeat(40)))
dxl_io.set_moving_speed(speed)
dxl_io.set_goal_position(dict(zip(ids, start_pose)))

name = input('Which gesture would you like to perform?\n')

found = 0
f=open("gesturelist.txt", "r")
fl=f.readlines()
for i in range((len(fl))):
	input = dict(eval(fl[i]))
	if input['name']=='{}'.format(name):
		found = 1
		f.close()
		all_ids = input['ids']
		all_poses = input['poses']
		all_times = input['times']
		perform(dxl_io, ids, all_ids, all_poses, all_times)
		break
if found == 0:
	print('Not Found')
dxl_io.disable_torque(ids)