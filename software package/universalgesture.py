import itertools
import time
import pypot.dynamixel

def perform(dxl_io, ids, all_ids, all_poses, all_times):
	dxl_io.enable_torque(ids)
	print('Preparing to perform gesture. Move away!\n')
	time.sleep(2)
	speed = dict(zip(ids, itertools.repeat(40)))
	dxl_io.set_moving_speed(speed)
	start_pose=[ -45, -9.53, 45.6, 30.06, -7.48, 49.12, -46.77, -44.13, -27.13, 8.94, -37.39, 50.0]
	move_to_pose(dxl_io, ids, dxl_io.get_present_position(ids), start_pose, 2)
	for i in range(len(all_ids)):
		begin_pose = dxl_io.get_present_position(all_ids[i])
		move_to_pose(dxl_io, all_ids[i], begin_pose, all_poses[i], all_times[i])
	time.sleep(1)
	move_to_pose(dxl_io, ids, dxl_io.get_present_position(ids), start_pose, 2)
	time.sleep(.5)
	#print('Press Ctrl-C to exit.')

def move_to_pose(dxl_io, ID, begin_pose, end_pose, t):
	distance = []
	speed = []
	for i in range(len(ID)):
		distance.append(abs(end_pose[i]-begin_pose[i]))
		motorspeed = distance[i]/t
		limit = 60
		if motorspeed < limit:
			speed.append(distance[i]/t)
		else:
			speed.append(limit)
	dxl_io.set_moving_speed(dict(zip(ID, speed)))
	dxl_io.set_goal_position(dict(zip(ID, end_pose)))
	time.sleep(t+.5)