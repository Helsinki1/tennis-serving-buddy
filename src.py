import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

camera = cv.VideoCapture(0)
if not camera.isOpened():
   print("ERROR: camera couldnt be opened")
   exit()

objectDetector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

tracker = cv.legacy.TrackerMOSSE()
success, img = camera.read()
tracker.init(img)

while True:

   captured, frame = camera.read()
   if not captured:
      print("ERROR: frame captured incorrectly, exiting...")
      break
   success, bbox = tracker.update(frame)
   x,y,w,h=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])

   # detect objects & draw objects
   mask = objectDetector.apply(frame)
   _, mask = cv.threshold(mask,254,255,cv.THRESH_BINARY)
   contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   
   blankFrame = np.copy(frame) * 0
   for cnt in contours:
      # remove static & small elements
      area = cv.contourArea(cnt)
      if area > 20:
         cv.drawContours(blankFrame, [cnt], -1, (0,255,0), 2)


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
