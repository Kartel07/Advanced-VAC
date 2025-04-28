import cv2
import numpy as np
import datetime
offset = 6
ya = 241
yb = 68
yc = 68
yd = 270
xa = 29
xb = 352
xc = 610
xd = 561
area = [(xa,ya),(xb,yb),(xc,yc),(xd,yd)]
detect = []
counter_list = []
cap = cv2.VideoCapture('video_new.mp4')

#Helps get point
def rgb (event,xx,yy,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [xx,yy]
        print(point)


cv2.namedWindow("FRAME")
cv2.setMouseCallback("FRAME", rgb)


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




#Creates frame for video
def frame_maker():
    # Web Cam
    ret, frame = cap.read()
    return frame


def counter_vehicle():
    counter = 0
    start_time = datetime.datetime.now()
    # end time is 10 sec after the current time
    end_time = start_time + datetime.timedelta(seconds=16)
    # Run the loop till current time exceeds end time
    while end_time > datetime.datetime.now():
        frame = frame_maker()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Converts the color 'frame' to a grayscale image using the OpenCV function 'cv2.cvtColor()' and stores it in the 'grey' variable. Grayscale images are often used for object detection as they reduce complexity.
        blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Applies a Gaussian blur to the  image. This helps to reduce noise and smooth the image, making object detection more robust. The '(3, 3)' is the kernel size for the blur, and '5' is the standard deviation in the X and Y directions.
        area_rect = cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2) #draws and area of detection
        # Applying on all frames
        img_sub = algo.apply(blur)# Applies a background subtraction algorithm to the blurred grayscale image. This aims to isolate moving objects  by subtracting the static background.
        dilate = cv2.dilate(img_sub, np.ones((5, 5))) # Dilation expands the bright regions in the image, which can help to fill in gaps in detected objects and connect nearby parts. 'np.ones((5, 5))' creates a 5x5 kernel of ones, which is used for the dilation.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Creates an elliptical structuring element. This kernel shape is often used in morphological operations for object analysis. The size is 5x5.
        dilate_data = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)  # Applies a morphological closing operation  image using the kernel. Closing is a combination of dilation followed by erosion and helps to fill small holes within objects and connect nearby objects.
        dilate_data = cv2.morphologyEx(dilate_data, cv2.MORPH_CLOSE, kernel) # Applies another morphological closing operation to the image. This second pass further refines the detected object shapes.
        counter_shape, h = cv2.findContours(dilate_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Finds contours (boundaries of connected white pixels) in the image using 'cv2.findContours()'.
        # 'cv2.RETR_TREE' retrieves all the contours and reconstructs a full hierarchy of nested contours.
        # 'cv2.CHAIN_APPROX_SIMPLE' compresses horizontal, vertical, and diagonal segments into their end points, reducing the number of points needed to represent the contour.
        # The function returns the contours themselves ('counter_shape') and a hierarchy of the contours ('h').

        for (i, c) in enumerate(counter_shape):
            # Iterates through each contour 'c' found in 'counter_shape'. 'enumerate()' provides both the index 'i' and the contour 'c'.
            (x, y, w, h) = cv2.boundingRect(c)
            # Calculates the bounding rectangle (x, y coordinates of the top-left corner, width 'w', and height 'h') for the current contour 'c' using 'cv2.boundingRect()'.
            val_counter = (w >= min_width_rectangle) and (h >= min_height_rectangle)
            # Checks if both the width 'w' and height 'h' of the bounding rectangle are greater than or equal to the predefined minimum width and height for a vehicle. The result (True or False) is stored in 'val_counter'.
            if not val_counter:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(frame, "Vehicle: " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            for (x, y) in detect:
                # if (yb - offset - 6) < y < (yb + offset + 6):
                #     counter += 1
                if ya-offset<=y<=ya+offset:
                    counter += 1
                detect.remove((x, y))

        cv2.putText(frame, "Counter: " + str(counter), (200, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
        counter_list.append(counter)
        cv2.imshow('FRAME', frame)
        if cv2.waitKey(12) == 13:
            break
    cv2.destroyAllWindows()
    cap.release()
    return counter_list[-1]

def main():
    counter_vehicle()


if __name__ == "__main__":
    main()