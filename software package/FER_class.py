import cv2
import glob
import random
import dlib
import numpy as np
import math
import itertools
from sklearn.svm import SVC
import PIL
from PIL import Image
from sklearn.externals import joblib
import time


class start_FER:

    def get_landmarks(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1)
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
            xlist = []
            ylist = []
            landmarks= []
            for i in range(0,68): #Store X and Y coordinates in two lists
                cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 
                #For each point, draw a red circle with thickness2 on the original frame
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist) #Find both coordinates of centre of gravity
            ymean = np.mean(ylist)
            x_max = np.max(xlist)
            x_min = np.min(xlist)
            y_max = np.max(ylist)
            y_min = np.min(ylist)
            cv2.rectangle(clahe_image,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,150,0),2)
            # print ("centre of gravity",xmean, ymean)
            # print ("range of the face",x_max, x_min, y_max, y_min)
            cv2.circle(clahe_image, (int(xmean), int(ymean) ), 1, (0,255,255), thickness=2) 
            x_start = int(x_min-5)
            y_start = int(y_min-((ymean - y_min)/3))
            w = int(x_max+5) - x_start
            h = int(y_max+5) - y_start

            xlist[:] = [x-x_start for x in xlist]
            ylist[:] = [y-y_start for y in ylist]

            xlist = np.array(xlist,dtype = np.float64)
            ylist = np.array(ylist,dtype = np.float64)
            # xlist = np.float32(xlist)
            # ylist = np.float32(ylist)

        if len(detections) > 0:
            return xlist, ylist
        else: #If no faces are detected, return error message to other function to handle
            xlist = np.array([])
            ylist = np.array([])
            return xlist, ylist


    def linear_interpolation(self,xlist,ylist):
        xlist = np.array(xlist,dtype = np.float64)
        ylist = np.array(ylist,dtype = np.float64)
        x_new = np.array([])
        y_new = np.array([])
        x = np.array([])
        y = np.array([])
        for i in range (len(xlist)-1):
            x_new = np.concatenate((x_new,[(xlist[i]+xlist[i+1])/2.0]))
            y_new = np.concatenate((y_new,[(ylist[i]+ylist[i+1])/2.0]))

        for j in range (len(xlist)):
            if j<(len(xlist)-1):
                x = np.concatenate((x,[xlist[j]]))
                x = np.concatenate((x,[x_new[j]]))
                y = np.concatenate((y,[ylist[j]]))
                y = np.concatenate((y,[y_new[j]]))
            else:
                x = np.concatenate((x,[xlist[j]]))
                y = np.concatenate((y,[ylist[j]]))
        return x, y


    def extract_AU(self,xlist,ylist):
        AU_feature = []
        Norm_AU_feature = []
        AU1_1_x = xlist[19:22]
        AU1_1_y = ylist[19:22]
        AU1_1_x,AU1_1_y = self.linear_interpolation(AU1_1_x,AU1_1_y)
        AU1_1_x,AU1_1_y = self.linear_interpolation(AU1_1_x,AU1_1_y)
        AU_feature = self.get_average_curvature(AU1_1_x,AU1_1_y)

        AU1_2_x = xlist[22:25]
        AU1_2_y = ylist[22:25]
        AU1_2_x,AU1_2_y = self.linear_interpolation(AU1_2_x,AU1_2_y)
        AU1_2_x,AU1_2_y = self.linear_interpolation(AU1_2_x,AU1_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU1_2_x,AU1_2_y)

        AU2_1_x = xlist[17:20]
        AU2_1_y = ylist[17:20]
        AU2_1_x,AU2_1_y = self.linear_interpolation(AU2_1_x,AU2_1_y)
        AU2_1_x,AU2_1_y = self.linear_interpolation(AU2_1_x,AU2_1_y)
        AU_feature = AU_feature + self.get_average_curvature(AU2_1_x,AU2_1_y)
        AU2_2_x = xlist[24:27]
        AU2_2_y = ylist[24:27]
        AU2_2_x,AU2_2_y = self.linear_interpolation(AU2_2_x,AU2_2_y)
        AU2_2_x,AU2_2_y = self.linear_interpolation(AU2_2_x,AU2_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU2_2_x,AU2_2_y)

        AU5_1_x = xlist[36:40]
        AU5_1_y = ylist[36:40]
        AU5_1_x,AU5_1_y = self.linear_interpolation(AU5_1_x,AU5_1_y)
        AU5_1_x,AU5_1_y = self.linear_interpolation(AU5_1_x,AU5_1_y)
        AU_feature = AU_feature +self.get_average_curvature(AU5_1_x,AU5_1_y)
        AU5_2_x = xlist[42:46]
        AU5_2_y = ylist[42:46]
        AU5_2_x,AU5_2_y = self.linear_interpolation(AU5_2_x,AU5_2_y)
        AU5_2_x,AU5_2_y = self.linear_interpolation(AU5_2_x,AU5_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU5_2_x,AU5_2_y)

        AU7_1_x = np.append(xlist[39:42],xlist[36])
        AU7_1_y = np.append(ylist[39:42],ylist[36])
        AU7_1_x,AU7_1_y = self.linear_interpolation(AU7_1_x,AU7_1_y)
        AU7_1_x,AU7_1_y = self.linear_interpolation(AU7_1_x,AU7_1_y)
        AU_feature = AU_feature + self.get_average_curvature(AU7_1_x,AU7_1_y)

        AU7_2_x = np.append(xlist[46:48],xlist[42])
        AU7_2_y = np.append(ylist[46:48],ylist[42])
        AU7_2_x,AU7_2_y = self.linear_interpolation(AU7_2_x,AU7_2_y)
        AU7_2_x,AU7_2_y = self.linear_interpolation(AU7_2_x,AU7_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU7_2_x,AU7_2_y)

        AU9_x = xlist[31:36]
        AU9_y = ylist[31:36]
        AU9_x,AU9_y = self.linear_interpolation(AU9_x,AU9_y)
        AU9_x,AU9_y = self.linear_interpolation(AU9_x,AU9_y)
        AU_feature = AU_feature + self.get_average_curvature(AU9_x,AU9_y)

        AU10_x = np.append(xlist[48:51],xlist[52:55])
        AU10_y = np.append(ylist[48:51],ylist[52:55])
        AU10_x,AU10_y = self.linear_interpolation(AU10_x,AU10_y)
        AU10_x,AU10_y = self.linear_interpolation(AU10_x,AU10_y)
        AU_feature = AU_feature + self.get_average_curvature(AU10_x,AU10_y)

        AU12_1_x = [xlist[48]] + [xlist[60]] + [xlist[67]]
        AU12_1_y = [ylist[48]] + [ylist[60]] + [ylist[67]]
        AU12_1_x,AU12_1_y = self.linear_interpolation(AU12_1_x,AU12_1_y)
        AU12_1_x,AU12_1_y = self.linear_interpolation(AU12_1_x,AU12_1_y)
        AU_feature = AU_feature + self.get_average_curvature(AU12_1_x,AU12_1_y)

        AU12_2_x = [xlist[54]] + [xlist[64]] + [xlist[65]]
        AU12_2_y = [ylist[54]] + [ylist[64]] + [ylist[65]]
        AU12_2_x,AU12_2_y = self.linear_interpolation(AU12_2_x,AU12_2_y)
        AU12_2_x,AU12_2_y = self.linear_interpolation(AU12_2_x,AU12_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU12_2_x,AU12_2_y)


        AU20_x = xlist[55:60]
        AU20_y = ylist[55:60]
        AU20_x,AU20_y = self.linear_interpolation(AU20_x,AU20_y)
        AU20_x,AU20_y = self.linear_interpolation(AU20_x,AU20_y)
        AU_feature = AU_feature + self.get_average_curvature(AU20_x,AU20_y)

        Norm_AU_feature = (AU_feature-np.min(AU_feature))/np.ptp(AU_feature)



        return Norm_AU_feature


    def get_average_curvature(self,AU_xlist,AU_ylist):
        K = []
        Z = np.polyfit(AU_xlist,AU_ylist,3)
        P = np.poly1d(Z)
        P_1 = np.poly1d.deriv(P)
        P_2 = np.poly1d.deriv(P_1)
        for i in range(len(AU_xlist)):
            # K[i] =  P_2[AU_xlist[i]]/math.pow((1+math.pow((P_1(AU_xlist[i])),2)),1.5)
            Y = 1+math.pow(P_1(AU_xlist[i]),2)
            Y = math.pow(Y,1.5)
            # print("Y",Y)
            # print("X",P_2(AU_xlist[i]))
            K.append(P_2(AU_xlist[i])/Y)
        # m_K = np.mean(K)
        m_K = K
        return m_K


    def get_vectorized_landmark(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1)
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
            xlist = []
            ylist = []
            for i in range(0,68): #Store X and Y coordinates in two lists
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]
            landmarks_dist = []
            landmarks_theta = []
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                # landmarks_vectorized.append(w)
                # landmarks_vectorized.append(z)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)

                landmarks_dist.append(dist)
                landmarks_theta.append((math.atan2(y, x)*360)/(2*math.pi))

            landmarks_dist = landmarks_dist[17:]
            landmarks_theta = landmarks_theta[17:]
            landmarks_dist = np.array(landmarks_dist,dtype = np.float64)
            Norm_landmarks_dist = (landmarks_dist-np.min(landmarks_dist))/np.ptp(landmarks_dist)
            landmarks_theta = np.array(landmarks_theta,dtype = np.float64)
            Norm_landmarks_theta = (landmarks_theta-np.min(landmarks_theta))/np.ptp(landmarks_theta)

            landmarks_vectorized =  np.concatenate((Norm_landmarks_dist,Norm_landmarks_theta))

            # print("vectorized landmarks", landmarks_vectorized)
            return landmarks_vectorized
        if len(detections) < 1:
            landmarks_vectorized = np.array([])
        return landmarks_vectorized

    def __init__(self):
        # Real-time
        emotions = ["anger",  "disgust" ,"fear","happiness", "neutral", "sadness", "surprise"] #Emotion list

        w1 = 0.75
        w2 = 1-w1
        global detector
        global predictor

        video_capture = cv2.VideoCapture(0) #Webcam object
        detector = dlib.get_frontal_face_detector() #Face detector
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
        while True:
            realtime_data = np.array([])
            ret, frame = video_capture.read()
            # [xlist, ylist] = get_landmarks(frame)
            [xlist, ylist] = self.get_landmarks(frame)
            vec_landmark = self.get_vectorized_landmark(frame)*w1
            if (xlist.size) and (vec_landmark.size):
                Norm_AU_feature = self.extract_AU(xlist,ylist)*w2
                vec_AU = np.concatenate((Norm_AU_feature,vec_landmark))
                vec_AU = ((vec_AU-np.min(vec_AU))/np.ptp(vec_AU))
                realtime_data = np.concatenate((realtime_data,vec_AU))
                font = cv2.FONT_HERSHEY_SIMPLEX

            # print (realtime_data)

                clf = joblib.load('best_landmark_SVM.pkl') 
                Y = clf.predict([realtime_data])
                if Y == 0:
                    cv2.putText(frame,'anger',(50,70), font, 2,(0,0,255),3)
                if Y ==1:
                    cv2.putText(frame,'disgust',(50,70), font, 2,(0,0,255),3)
                if Y == 2:
                    cv2.putText(frame,'fear',(50,70), font, 2,(0,0,255),3)
                if Y ==3:
                    cv2.putText(frame,'happiness',(50,70), font, 2,(0,0,255),3)
                if Y==4:
                    cv2.putText(frame,'neutral',(50,70), font, 2,(0,0,255),3)
                if Y ==5:
                    cv2.putText(frame,'sadness',(50,70), font, 2,(0,0,255),3)
                if Y==6:
                    cv2.putText(frame,'surprise',(50,70), font, 2,(0,0,255),3)
            # cv2.namedWindow("Facial emotion recognition")        # Create a named window
            # cv2.moveWindow("Facial emotion recognition", 10,10)  # Move it to (40,30)
            cv2.imshow("Facial emotion recognition", frame) #Display the frame

            if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
                break