import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
#Helps get point
def rgb (event,xx,yy,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [xx,yy]
        print(point)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", rgb)

#loads YOLO v8 models
model = YOLO("yolo11s.pt")
names = model.names


# Basic variables
cap = cv2.VideoCapture("video.mp4")
count = 0
ya = 385
yb = 570
area = [(4,yb),(359,ya),(890,ya),(1182,yb)]
detect = []
counter = 0
offset = 6
while True:
     ret, frame = cap.read()
     if not ret:
         break
     count += 1
     if count %2 != 0:
         continue
     #frame = cv2.resize(frame,(1020,500))
     #Run YOLO tracking on frame
     result = model.track(frame,persist=True)
     #check if boxes in result
     area_rect = cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
     if result[0].boxes is not None and result[0].boxes.id is not None:
         #Get boxes
         boxes = result[0].boxes.xyxy.int().cpu().tolist() #binding boxes
         class_ids = result[0].boxes.cls.int().cpu().tolist() #class id
         track_ids = result[0].boxes.id.int().cpu().tolist() #track id
         confidences = result[0].boxes.conf.int().cpu().tolist() #confidence score
         for box, class_id,track_id,confidence in zip(boxes,class_ids,track_ids,confidences):
             c = names[class_id]
             if 'cars' or 'trucks' or 'buses' or 'motorcycles' in c:
                x1,y1,x2,y2 = box
                cx = int(x1+x2)//2
                cy = int(y1+y2)//2
                result = cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
                if result>=0:
                    cv2.circle(frame, (cx,cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)

     cv2.imshow("RGB", frame)
     if cv2.waitKey(1) == 13:
         break
cap.release()
cv2.destroyAllWindows()