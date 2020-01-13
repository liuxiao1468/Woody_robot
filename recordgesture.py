import itertools
import time
import pypot.dynamixel
from universalgesture import perform

# def perform(all_ids, all_poses, all_times):
# 	dxl_io.enable_torque(ids)
# 	print('Preparing to perform gesture. Move away!\n')
# 	time.sleep(2)
# 	speed = dict(zip(ids, itertools.repeat(40)))
# 	dxl_io.set_moving_speed(speed)
# 	start_pose=[ -45, -9.53, 45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]
# 	move_to_pose(ids, dxl_io.get_present_position(ids), start_pose, 1)
# 	time.sleep(3)
# 	for i in range(len(all_ids)):
# 		begin_pose = dxl_io.get_present_position(all_ids[i])
# 		move_to_pose(all_ids[i], begin_pose, all_poses[i], all_times[i])
# 	move_to_pose(ids, dxl_io.get_present_position(ids), start_pose, 1)
# 	print('Press Ctrl-C to exit.')
# 	while True:
# 		pass

# def move_to_pose(ID, begin_pose, end_pose, t):
# 	distance = []
# 	speed = []
# 	for i in range(len(ID)):
# 		distance.append(abs(end_pose[i]-begin_pose[i]))
# 		speed.append(distance[i]/t)
# 	dxl_io.set_moving_speed(dict(zip(ID, speed)))
# 	dxl_io.set_goal_position(dict(zip(ID, end_pose)))
# 	time.sleep(t+.5)

def beginrecording(dxl_io, ids, start_pose):
	dxl_io.enable_torque(ids)
	speed = dict(zip(ids, itertools.repeat(40)))
	dxl_io.set_moving_speed(speed)
	dxl_io.set_goal_position(dict(zip(ids, start_pose)))
	time.sleep(2)
	dxl_io.disable_torque(ids)

	all_ids = ()
	all_poses = ()
	all_times = []
	movetime = 0

	print('Please move to the first position\n')
	try:
		while True:
			match = False
			while True:
				for i in ids:
					if abs(start_pose[i-1]-dxl_io.get_present_position([i])[0])>5:
						match = True
						break
				if match == True:
					break
			print('motion detected')
			if movetime == 0:
				starttime = time.time()
			new_pose = dxl_io.get_present_position(ids)
			while True:
				match = False
				time.sleep(.1)
				check_pose = dxl_io.get_present_position(ids)
				for i in ids:
					if abs(new_pose[i-1]-check_pose[i-1])>5:
						match = True
						break
				new_pose = check_pose
				if match == False:
					break
			print('stop detected')
			changed_ids=[]
			changed_pose=[]
			for i in range(len(ids)):
				if abs(new_pose[i]-start_pose[i]) > 10:
					changed_ids.append(i+1)
					changed_pose.append(new_pose[i])
			if len(changed_ids) > 0:
				all_ids += (changed_ids,)
				all_poses += (changed_pose,)
				start_pose = new_pose
				movetime = time.time()-starttime
				all_times.append(movetime)
				movetime = 0
			else:
				movetime = 1
			print(all_ids)
			print(all_poses)
			print(all_times)
			print('Please move to the next position or press Ctrl-C to stop recording\n')
	except KeyboardInterrupt:
		pass

	try:
		name = input("Input name: \n")
		all_data = dict(zip(['name', 'ids', 'poses', 'times'], (name, all_ids, all_poses, all_times)))
		f=open('gesturelist.txt','a')
		f.write(('{}\n').format(all_data))
		f.close()
	except KeyboardInterrupt:
		dxl_io.disable_torque(ids)

# ports = pypot.dynamixel.get_available_ports()
# print('available ports:', ports)  

# if not ports:
# 	raise IOError('No port available.') 

# port = ports[0]
# print('Using the first on the list', port)

# dxl_io = pypot.dynamixel.DxlIO(port)
# print('Connected!')

# found_ids = dxl_io.scan(range(13))
# print('Found ids:', found_ids)

# if len(found_ids) < 2:
# 	raise IOError('You should connect at least two motors on the bus for this test.')
# #chose all motors and enable torque and set the same speed
# ids = found_ids[:]
# start_pose=[ -45, -9.53, 45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]

# beginrecording(dxl_io, ids, start_pose)