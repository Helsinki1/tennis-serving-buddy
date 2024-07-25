import FpsTracking  # class: FpsTracker
import CameraStream # class: CameraStream
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


fps = FpsTracking.FpsTracking()
stream = CameraStream.CameraStream()

objectDetector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

#for pyplot
ballx = []
bally = []
coordPairs = np.array([[]], ndmin=2)

input("Ready to capture first frame? ")
success, frame = stream.camera.read()
Fheight, Fwidth = frame.shape[0:2]

ans = input("Ready to select ROI? ")
croppedx, croppedy, croppedw, croppedh = cv.selectROI("select ROI", frame, showCrosshair=False) # allow user to manually exclude background
cv.destroyWindow("select ROI")
Fheight = croppedh
Fwidth = croppedw
stream.setROI(croppedx, croppedy, croppedw, croppedh)




stream.start()
fps.start()


while True:
   if stream.frameReady:
      frame = stream.returnFrame()

      # detect objects & draw objects
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
            cv.rectangle(blankFrame, (x, y), (x+w, y+h), (0, 0, 200), 3) # SUPPRESS IN FINAL PRODUCT
            ballx.append( Fwidth - (x + (w//2)) )
            bally.append( Fheight - (y + h) )

      # detect lines
      grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      t_lower = 50
      t_upper = 150
      L2Grad = True
      edgesFrame = cv.Canny(grayFrame, t_lower, t_upper, L2gradient = L2Grad)

      kernel = np.ones((5,5), np.uint8) # creates 3x3 array full of 1's in uint8
      edgesFrame = cv.dilate(edgesFrame, kernel, iterations=1) # enlarges white lines to merge both borders of a line into one edge
      kernel = np.ones((7,7), np.uint8)
      edgesFrame = cv.erode(edgesFrame, kernel, iterations=1) # thins white lines to merge edges so houghlines doesnt mark down too many lines

      rho = 1 # distance resolution
      theta = np.pi/180 # angular resolution
      threshold = 15 # min number of votes
      minLineLength = 50 # min pixels to be a line
      maxLineGap = 30 # max pixels between parts of a line
      lines = cv.HoughLinesP(edgesFrame, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

      #lines.append([0,0,Fwidth-1,0]) # in case the 4 vertices of baseline cant be captured bc of space permitted

      # draw lines
      for line in lines:
         for x1,y1,x2,y2 in line:
            cv.line(blankFrame, (x1,y1),(x2,y2),(255,0,0),5) # SUPPRESS IN FINAL PRODUCT

      cv.imshow('Lines & Objects', blankFrame) # SUPPRESS IN FINAL PRODUCT
      fps.updateCount()
      stream.frameReady = False

   if cv.waitKey(1) == ord('q'):
      stream.stop()
      fps.stop()
      break




fig, ax = plt.subplots()
ax.set_facecolor("yellowgreen")


# record all the corners created by court lines
grayFrame = np.float32(grayFrame)
dst = cv.cornerHarris(grayFrame, 2, 3, 0.04)  # find harris corners

kernel = np.ones((7,7), np.uint8)
dst = cv.dilate(dst, kernel)  # enlargen lighter areas and remove noise
#dst = cv.erode(dst, kernel)

_, dst = cv.threshold(dst,0.01*dst.max(),255,0)  # turn dst black and white
dst = np.uint8(dst)
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(grayFrame, np.float32(centroids), (5,5), (-1,-1), criteria)  # refines corner coords to find subpixel location
cv.imshow("grayscale + dilate + threshold for corners", dst)

# plot all detected corners
corners = np.intp(corners)
for [x,y] in corners[1:]:
   plt.plot(x, y, 'mo')


#draw the tennis court using patches
vertices = []
for [x,y] in corners[1:]:
   vertices.append([x,y]) # switch from numpy array to python array

def sortFirst(arr):
   return arr[0]
def sortSecond(arr):
   return arr[1]
vertices.sort(reverse=True, key=sortSecond) # baseline vertices first, then service, then net
baseLine = vertices[0:4]
baseLine.sort(key=sortFirst) # the four vertices of the baseline, L to R
serviceLine = vertices[4:7]
serviceLine.sort(key=sortFirst) # the three vertices of serviceline, L to R
netLine = vertices[7:]
netLine.sort(key=sortFirst) # the seven vertices of the netline, L to R

courtTL = (baseLine[0][0], baseLine[0][1])
courtTR = (baseLine[3][0], baseLine[3][1])
courtBR = (netLine[5][0], netLine[5][1])
courtBL = (netLine[1][0], netLine[1][1])
serveVertices = np.array([courtTL, courtTR, courtBR, courtBL]) # PLACEHOLDER CONSTANTS
shape = patches.Polygon(serveVertices, color="cornflowerblue")
ax.add_patch(shape)



# take a snapshot of the court lines on the last captured frame
numLines = len(lines)
coordPairs = [[0,0,0,0] for i in range(numLines)]
ind = 0
for line in lines:
   for x1,y1,x2,y2 in line:
      coordPairs[ind] = [x1,y1,x2,y2]
      ind += 1

#graph all detected court lines
for [x1,y1,x2,y2] in coordPairs:
   ax.plot([x1,x2],[y1,y2], color='white', linestyle='solid', linewidth=2)
   plt.plot(x1,y1, color='m')
   plt.plot(x2,y2, color='m')
   #plt.axline((x1,y1),(x2,y2), linewidth=2, color='w')
   #plt.plot([x1,x2],[y1,y2], 'w', linestyle="solid")

#graph ball trajectory
plt.plot(ballx, bally, 'ro')
plt.axis((0,Fwidth,0,Fheight))

#plot lowest point of ball traj (the bounce)
index = bally.index(min(bally))
bounce = [ballx[index], bally[index]]
plt.plot(bounce[0], bounce[1], 'go')
plt.show()

#conclude by exiting from everything
stream.camera.release
cv.destroyAllWindows()

print("Avg FPS:", int(fps.compute()))