import cv2, sys, os, numpy

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

	# cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))
	#cascadeL = cv2.CascadeClassifier("C:/Users/david/Desktop/MAG FRI/SB/Regular Track - Files for Assignment 2/detectors/cascade_detector/cascades/haarcascade_mcs_leftear.xml")
	#cascadeR = cv2.CascadeClassifier("C:/Users/david/Desktop/MAG FRI/SB/Regular Track - Files for Assignment 2/detectors/cascade_detector/cascades/haarcascade_mcs_rightear.xml")
	# cascade = cv2.CascadeClassifier('C:/Users/david/Desktop/MAG FRI/SB/Regular Track - Files for Assignment 2/detectors/cascade.xml')
	cascade = cv2.CascadeClassifier('C:/Users/david/Desktop/MAG FRI/SB/Regular Track - Files for Assignment 2/detectors/habit_haar_cascade_horse_ears_1.xml')

	def detect(self, img):
		try:
			det_list1 = self.cascadeR.detectMultiScale(img, 1.05, 1)
			det_list2 = self.cascadeL.detectMultiScale(img, 1.05, 1)
			if det_list1 == () and det_list2 == (): return ()
			elif det_list1 == ():return det_list2
			elif det_list2 == (): return det_list1
			else:
				return numpy.vstack([det_list1, det_list2])
		except Exception as e:
			print(e)

	def detect1(self, img):
		return self.cascade.detectMultiScale(img, 1.05, 1)