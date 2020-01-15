import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Gdk, GdkPixbuf
import cv2
import universalgesture
import recordgesture
import itertools
import time
import pypot.dynamixel

class Dialogtest(Gtk.Dialog):

    def __init__(self, parent):
        Gtk.Dialog.__init__(self, "Warning", parent, 0,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OK, Gtk.ResponseType.OK))

        self.set_default_size(300, 100)

        label = Gtk.Label("Your Woody is not connected correctly")

        box = self.get_content_area()
        box.add(label)
        self.show_all()

class MainWindow(Gtk.Window):
	def __init__(self):
		Gtk.Window.__init__(self, title="Woody")
		self.set_border_width(10)
		self.set_size_request(200, 100)
		#layout
		vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
		self.add(vbox)

		#add ecube image
		self.img = Gtk.Image.new_from_file('FER.png')
		vbox.pack_start(self.img, True, True, 0)

		#username
		#sign in button
		self.button = Gtk.Button(label = "Read Woody instructions")
		self.button.connect("clicked",self.instruction)
		vbox.pack_start(self.button, True, True, 0)


		#sign in button
		self.button = Gtk.Button(label = "Test your Woody")
		self.button.connect("clicked",self.test)
		vbox.pack_start(self.button, True, True, 0)

		#sign in button
		self.button = Gtk.Button(label = "Control your Woody")
		self.button.connect("clicked",self.control)
		vbox.pack_start(self.button, True, True, 0)


	def instruction(self, widget):
		window.hide()
		#SHOW PDF


	def test(self, widget):
		dialog = Dialogtest(self)
		response = dialog.run()

		if response == Gtk.ResponseType.OK:
			print("The OK button was clicked")
		elif response == Gtk.ResponseType.CANCEL:
			print("The Cancel button was clicked")

		dialog.destroy()


	def control(self, widget):
		window.hide()
		self.win = Gtk.Window(title="win")
		self.win.set_border_width(10)
		self.win.set_size_request(200, 520)
		#layout
		vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
		self.win.add(vbox)
		self.win.button = Gtk.Button(label = "Record New Gesture")
		vbox.pack_start(self.win.button, True, True, 0)
		self.win.button.connect("clicked", self.record, (dxl_io, ids, start_pose))
		f=open('gesturelist.txt','r')
		fl=f.readlines()
		for i in range((len(fl))):
			gesturedata = dict(eval(fl[i]))
			self.win.button = Gtk.Button(label=gesturedata['name'])
			vbox.pack_start(self.win.button, True, True, 0)
			all_ids = gesturedata['ids']
			all_poses = gesturedata['poses']
			all_times = gesturedata['times']

			self.win.button.connect("clicked", self.perform, (dxl_io, ids, all_ids, all_poses, all_times))
		f.close()

		self.win.connect("destroy", Gtk.main_quit)
		self.win.show_all()


	def back_to_main(self,widget):
		self.win2.hide()
		self.win.show()

	def record(self, widget, data):
		self.win.hide()
		self.winrecord = Gtk.Window(title = "Recording")
		self.winrecord.set_border_width(10)
		self.winrecord.set_size_request(200,520)
		self.winrecord.button = Gtk.Button(label='name')
		# vbox2 = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
		# self.winrecord.add(vbox2)
		# self.winrecord.button = Gtk.Button(label = "Record New Gesture")
		# vbox2.pack_start(self.winrecord.button, True, True, 0)
		self.winrecord.show_all()
		# Gtk.main()
		print(data)
		start_pose = data[2]

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

	def perform(self, widget, data):#self, dxl_io, ids, all_ids, all_poses, all_times):
		self.win.hide()
		print(data)
		universalgesture.perform(data[0],data[1],data[2],data[3],data[4])
		dxl_io.disable_torque(ids)
		self.win.show()

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

window = MainWindow()
window.connect("delete-event", Gtk.main_quit)
window.show_all()
Gtk.main()
