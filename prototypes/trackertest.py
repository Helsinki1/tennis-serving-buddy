import cv2

cap=cv2.VideoCapture(0)
success, img = cap.read()

tracker=cv2.legacy.TrackerMOSSE_create()
bbox=cv2.selectROI("output",img,False)
tracker.init(img ,bbox)

def draw(img,bbox):
    x,y,w,h=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,255),3,1)

while True:
    success,img=cap.read()
    sucess,bbox=tracker.update(img)
    if sucess:
        draw(img,bbox)
    else:
        cv2.putText(img,"lost",(75,55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv2.imshow("output",img)
    k=cv2.waitKey(1)


    if cv2.waitKey(1) == ord('q'):
        break

#conclude by exiting from everything
cap.release
cv2.destroyAllWindows()
