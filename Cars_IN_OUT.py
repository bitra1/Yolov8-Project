
from ultralytics import YOLO   # Imports the YOLO (You Only Look Once) model from the ultralytics library for object detection.
import cv2  #used for image processing and computer vision tasks.
import cvzone  #for displaying detection boxes and text
import math
from sort import *  #SORT tracking algorithm to track detected objects across frames
import numpy as np  #handling arrays

cap = cv2.VideoCapture("../Videos/c2_11.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")   #Loads the YOLO model weights from a file. This model is used to detect objects in the video.

with open ('coco.names','r') as f:   #list of class names (e.g., car, bus, etc.)
    ClassNames = f.read().splitlines()   #Reads the file line by line and splits it into a list of class names.

#Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)   #Initializes the SORT tracker, which tracks objects between frames

limits_out = [227,368,542,367]   # coordinates of the lines for counting objects moving "in" or "out" of the region of interest.
totalcount_out= [ ]  #Lists to keep track of the IDs of objects

limits_in = [ 231,271,536,270]
totalcount_in=[ ]

while True:
    ret,frame = cap.read()   #Reads the next frame from the video
    if not ret:     #Exits the loop when the video ends.
        break

    mask = np.zeros(frame.shape[:2], dtype='uint8')     #Creates a blank mask the same size as the frame
    cv2.rectangle(mask, (226, 257), (542, 401), (255, 255, 255), -1)   #Draws a white rectangle on the mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.polylines(frame, [np.array([[226, 257], [542, 257], [542, 403], [227, 402]], np.int32)], isClosed=True,
                  color=(0, 255, 0), thickness=2)   #Draws a green polygon on the frame around the area
    #cv2.polylines(frame, [np.array([[287,188],[569,188],[569,324],[286,324]], np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    result = model(masked_frame,stream=True)   #Runs the YOLO model on the masked frame, detecting objects.

    detections = np.empty((0,5))   #Creates an empty NumPy array to store the detection results (bounding box coordinates, confidence score).

    for r in result:
        boxes = r.boxes

        #bounding box
        for box in boxes:
            x1,y1,x2,y2 =box.xyxy[0]   #Extracts the bounding box coordinates for each detected object.
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            w,h = x2-x1,y2-y1

            # confidence
            conf = math.ceil((box.conf[0]*100))/100 #it's 2 Decimal Points we use multiple with 100 and divide by 100
            # print(conf)
            #cvzone.putTextRect(frame,f'{conf}',(max(0,x1),max(35,y1)))

            #class name
            cls = int(box.cls[0])   #Extracts the class index of the detected object.
            CurrentClass = ClassNames[cls]    #Maps the class index to the class name (e.g., "car").
            if (CurrentClass in ['car', 'truck', 'bicycle', 'motorbike', 'bus']) and conf > 0.3:
                #cvzone.putTextRect(frame,f'{CurrentClass} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1,offset=3)
                #cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=5)
                currentArray =np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))   #Adds the detection to the detections array.
    resultTracker = tracker.update(detections)
    # cv2.line(frame,(limits_out[0],limits_out[1]),(limits_out[2],limits_out[3]),(255,0,0),1)
    # cv2.line(frame,(limits_in[0],limits_in[1]),(limits_in[2],limits_in[3]),(255,0,0),1)
    for result in resultTracker:    #Updates the tracker with the new detections from the current frame.
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2,colorR=(255,0,255))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)   #Draws a small circle at the center of the object.


        cx,cy = x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),0,(255,0,255),cv2.FILLED)



        if limits_out[0] < cx < limits_out[2] and limits_out[1] - 12 < cy < limits_out[1] + 12 :  #checks if the object has crossed the "out" line, and adds the object's ID to totalcount_out if it hasn't already been counted.
            if totalcount_out.count(id) ==0:
                totalcount_out .append(id)
                # cv2.line(frame, (limits_out[0], limits_out[1]), (limits_out[2], limits_out[3]), (0, 0, 255), 2)

        if limits_in[0] < cx < limits_in[2] and limits_in[1] - 12 < cy < limits_in[1] + 12 :
            if totalcount_in.count(id) == 0:
                totalcount_in.append(id)
                # cv2.line(frame, (limits_in[0], limits_in[1]), (limits_in[2], limits_in[3]), (0, 0, 255), 2)
    cv2.putText(frame, f'In: {len(totalcount_in)}', (25, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(frame, f'Out: {len(totalcount_out)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow('Video_Capture',frame)
    cv2.imshow('Masking',masked_frame)


    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
