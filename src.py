import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

camera = cv.VideoCapture(0)
if not camera.isOpened():
   print("ERROR: camera couldnt be opened")
   exit()

objectDetector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

#for pyplot
ballx = []
bally = []
serveBoxEdge1 = [0,0,0,0] # far wide boundary x1x2y1y2
serveBoxEdge2 = [0,0,0,0] # long boundary x1x2y1y2
serveBoxEdge3 = [0,0,0,0] # near wide boundary x1x2y1y2
success, frame = camera.read()
Fheight, Fwidth = frame.shape[0:2]

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
      # draw boxes around all the detected objects & record ball coords
      area = cv.contourArea(cnt)
      if area > 35:
         x, y, w, h = cv.boundingRect(cnt)
         cv.rectangle(blankFrame, (x, y), (x+w, y+h), (0, 0, 200), 3)
         ballx.append( Fwidth - (x + (w//2)) )
         bally.append( Fheight - (y + h) )


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
         slope = (y2 - y1) / (x2 - x1)
         if (slope > 3) or ((-1*slope) < -3):
            serveBoxEdge2 = [Fwidth-x1,Fwidth-x2,Fheight-y1,Fheight-y2] # changing origin to bottom left corner
         else:
            if y1 < Fheight/2 and y2 < Fheight/2: # extremely rudimentary classification system
               serveBoxEdge1 = [Fwidth-x1,Fwidth-x2,Fheight-y1,Fheight-y2]
            else:
               serveBoxEdge2 = [Fwidth-x1,Fwidth-x2,Fheight-y1,Fheight-y2]


   cv.imshow('Lines & Objects',blankFrame)


   if cv.waitKey(1) == ord('q'):
      break


#graph ball trajectory
plt.plot(ballx, bally, 'ro')
plt.axis((0,Fwidth,0,Fheight))
plt.show()

#find lowest point of ball traj (the bounce)
index = bally.index(min(bally))
bounce = [ballx[index], bally[index]]

#conclude by exiting from everything
camera.release
cv.destroyAllWindows()
