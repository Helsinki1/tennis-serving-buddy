import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

camera = cv.VideoCapture(0)
if not camera.isOpened():
   print("ERROR: camera couldnt be opened")
   exit()

objectDetector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

while True:

   captured, frame = camera.read()
   if not captured:
      print("ERROR: frame captured incorrectly, exiting...")
      break

   # detect objects & draw objects (MIGHT BE REDUNDANT BC OBJ TRACKER)
   mask = objectDetector.apply(frame)
   _, mask = cv.threshold(mask,254,255,cv.THRESH_BINARY)
   contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   
   blankFrame = np.copy(frame) * 0
   for cnt in contours:
      # remove static & small elements
      # draw boxes around all the detected objects
      area = cv.contourArea(cnt)
      if area > 35:
         x, y, w, h = cv.boundingRect(cnt)
         cv.rectangle(blankFrame, (x, y), (x+w, y+h), (0, 0, 200), 3)
         ballx = x + (w//2)
         bally = y + h


   # detect lines
   grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

   t_lower = 50
   t_upper = 150
   L2Grad = True
   edgesFrame = cv.Canny(grayFrame, t_lower, t_upper, L2gradient = L2Grad)

   rho = 1 # distance resolution
   theta = np.pi/180 # angular resolution
   threshold = 15 # min number of votes
   minLineLength = 50 # min pixels to be a line
   maxLineGap = 30 # max pixels between parts of a line
   lines = cv.HoughLinesP(edgesFrame, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

   # draw lines
   for line in lines:
      for x1,y1,x2,y2 in line:
         cv.line(blankFrame, (x1,y1),(x2,y2),(255,0,0),5)


   cv.imshow('Lines & Objects',blankFrame)



   if cv.waitKey(1) == ord('q'):
      break

#conclude by exiting from everything
camera.release
cv.destroyAllWindows()
