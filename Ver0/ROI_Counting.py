import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import pickle
#Helps get point
def rgb (event,xx,yy,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [xx,yy]
        print(point)

def center_handle(x_co,y_co,wi,hei):
    x1 = int(wi/2)
    y1 = int(hei/2)
    cx = x_co+x1
    cy = y_co+y1
    return cx,cy


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
point_list = []
counter = 0
offset = 6
#initialising Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
#min width and min height

min_width_rectangle = 80
min_height_rectangle = 80


while True:
     ret, frame = cap.read()
     #Run YOLO tracking on frame
     result = model.track(frame,persist=True)
     # #check if boxes in result
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
                     point_list.append((cx,cy))
                     result = cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
                     if result>=0:
                         cv2.circle(frame, (cx,cy), 4, (0, 0, 255), -1)
                         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                         cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                         cv2.line(frame, (4, yb), (1182, yb), (0, 125, 0), 3)
                         if track_id in area_rect:
                             if detect.count(track_id) == 0:
                                 detect.append(track_id)
                                 counter += 1
                                 print("COUNT=", counter)
                                 if track_id not in area_rect:
                                     detect.remove(track_id)
                                     counter -= 1
                     cv2.putText(frame, "Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255),5)

     cv2.imshow("RGB", frame)
     if cv2.waitKey(1) == 13:
         break
cap.release()
cv2.destroyAllWindows()
