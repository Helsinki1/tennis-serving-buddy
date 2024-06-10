import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

camera = cv.VideoCapture(0);
if not camera.isOpened():
   print("ERROR: camera couldnt be opened")
   exit()
while True:
   captured, frame = camera.read()
   if not captured:
      print("ERROR: frame captured incorrectly, exiting...")
      break
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
   frameCopy = np.copy(frame) * 0
   for line in lines:
      for x1,y1,x2,y2 in line:
         cv.line(frameCopy, (x1,y1),(x2,y2),(255,0,0),5)
         frameWithEdges = cv.addWeighted(frame, 0.8, frameCopy, 1, 0)
         cv.imshow('lines',frameWithEdges)

   if cv.waitKey(1) == ord('q'):
      break

#conclude by exiting from everything
camera.release
cv.destroyAllWindows()
