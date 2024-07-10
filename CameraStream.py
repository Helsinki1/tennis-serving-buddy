from threading import Thread
import cv2 as cv

class CameraStream:

    def __init__(self, src=0): # if no camera src reported, assume default webcam
        self.camera = cv.VideoCapture(src)
        if not self.camera.isOpened():
            print("ERROR: camera couldnt be opened")
            exit()
        self.stopThreads = False
        self.useROI = False
        self.frameReady = False
        self.cropped = None


    def setROI(self, x, y ,w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.useROI = True


    def start(self):
        Thread(target=self.run, args=()).start()


    def run(self):
        while not self.stopThreads:
            self.success, self.frame = self.camera.read()
            if not self.success:
                print("ERROR: frame captured incorrectly, exiting...")
                return
            if self.useROI:
                self.cropped = self.frame[(self.x):(self.x+self.w), (self.y):(self.y+self.h)]
            self.frameReady = True
        

    def returnFrame(self):
        if self.useROI:
            return self.cropped
        else:
            return self.frame


    def stop(self):
        self.stopThreads = True