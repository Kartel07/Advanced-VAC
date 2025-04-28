import cv2
import numpy as np

#Web Cam
cap = cv2.VideoCapture('video.mp4')

#Line Counter

line_pos = 550
offset = 6
counter = 0

#min width and min height

min_width_rectangle = 80
min_height_rectangle = 80

#initialising Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def center_handle(x_co,y_co,wi,hei):
    x1 = int(wi/2)
    y1 = int(hei/2)
    cx = x_co+x1
    cy = y_co+y1
    return cx,cy

detect = []

while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #Applying on all frames
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilate_data = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    dilate_data = cv2.morphologyEx(dilate_data, cv2.MORPH_CLOSE, kernel)
    counter_shape,h = cv2.findContours(dilate_data, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1,(25,line_pos),(1200,line_pos),(255,127,0),3)

    for  (i,c) in enumerate(counter_shape):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = (w>=min_width_rectangle) and (h>=min_height_rectangle)
        if not val_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1, "Vehicle: " + str(counter), (x, y-20), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)


        for (x,y) in detect:
            if (line_pos - offset) < y < (line_pos + offset):
                counter += 1
            cv2.line(frame1,(25,line_pos),(1200,line_pos),(0,125,0),3)
            detect.remove((x,y))
            print("Counter: "+str(counter))



    cv2.putText(frame1,"Counter: "+str(counter),(450,70),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),5)

    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()