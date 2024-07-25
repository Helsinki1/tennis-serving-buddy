import datetime

class FpsTracking:
    def __init__(self):
        self.startTime = None # set vars for datetime data type to null
        self.endTime = None
        self.numFrames = 0
    
    def start(self):
        self.startTime = datetime.datetime.now()

    def stop(self):
        self.endTime = datetime.datetime.now()

    def updateCount(self):
        self.numFrames += 1

    def compute(self):
        return self.numFrames / (self.endTime - self.startTime).total_seconds()