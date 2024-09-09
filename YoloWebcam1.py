import numpy as np
from ultralytics import YOLO
import cv2
import cvzone  #display detection
import math
from sort import *

#cap = cv2.VideoCapture(0)
# cap.set(3,1200)
# cap.set(4,720)

cap = cv2.VideoCapture("../Videos/c2_11.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

with open ('coco.names','r') as f:
    ClassNames = f.read().splitlines()

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
line = [266,333,444,428]
totalcount=[ ]

while True:
    ret,frame = cap.read()
    mask = np.zeros(frame.shape[:2],dtype='uint8')
    cv2.fillPoly(mask, [np.array([[343,255],[516,323],[435,445],[256,345]],np.int32)], (255,0,0))
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.polylines(frame, [np.array([[343,255],[516,323],[435,445],[256,345]], np.int32)], isClosed=True,
                  color=(0, 255, 0), thickness=2)

    result = model(masked_frame)
    detections = np.empty((0, 5))
    for r in result:
        boxes = r.boxes

        #bounding box
        for box in boxes:
            x1,y1,x2,y2 =box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            # w,h = x2-x1,y2-y1
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            # cvzone.cornerRect(frame,(x1,y1,w,h))
            #print('The Box value=',box)

            # confidence
            conf = math.ceil((box.conf[0]*100))/100 #it's 2 Decimal Points we use multiple with 100 and divide by 100
            # print(conf)
            #cvzone.putTextRect(frame,f'{conf}',(max(0,x1),max(35,y1)))

            #class name
            cls = int(box.cls[0])
            currentClass = ClassNames[cls]
            if(currentClass in ['car', 'truck', 'bicycle', 'motorbike', 'bus']) and conf > 0.3:
                # Display class name and confidence on the frame
                cv2.putText(frame, f'{currentClass} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                cv2.rectangle(frame, (x2, y2), (x1, y1), (0, 0, 255), 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections,currentArray))


    resultsTracker = tracker.update(detections)
    cv2.line(frame,(line[0],line[1]),(line[2],line[3]),(255,0,0),2)
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(frame, (x1, y1),(w, h),  (255, 0, 255), 2)
        cv2.putText(frame, f' {int(id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


        cx,cy = x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
        if line[0] < cx < line[2] and line[1] - 15 < cy < line[1] + 15:
            if totalcount.count(id) == 0:
                totalcount.append(id)
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

    cv2.putText(frame, f'Count  : {len(totalcount)}', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow('Video_Capture',frame)
    cv2.imshow('Masked_Frame',masked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



#
# from ultralytics import YOLO
# import cv2
# import cvzone  #display detection
# import math
# from sort import *
# import numpy as np
#
# cap = cv2.VideoCapture("../Videos/c2_11.mp4")
#
# model = YOLO("../Yolo-Weights/yolov8s.pt")
#
# with open ('coco.names','r') as f:
#     ClassNames = f.read().splitlines()
#
# #Tracking
# tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
#
# limits = [227,368,542,367]
# totalcount = [ ]
#
# while True:
#     ret,frame = cap.read()
#     if not ret:
#         break
#
#     mask = np.zeros(frame.shape[:2], dtype='uint8')
#     cv2.rectangle(mask, (226, 257), (542, 401), (255, 255, 255), -1)
#     masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.polylines(frame, [np.array([[226, 257], [542, 257], [542, 403], [227, 402]], np.int32)], isClosed=True,
#                   color=(0, 255, 0), thickness=2)
#     #cv2.polylines(frame, [np.array([[287,188],[569,188],[569,324],[286,324]], np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
#     result = model(masked_frame,stream=True)
#
#     detections = np.empty((0,5))
#
#     for r in result:
#         boxes = r.boxes
#
#         #bounding box
#         for box in boxes:
#             x1,y1,x2,y2 =box.xyxy[0]
#             x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
#             #print(x1,y1,x2,y2)
#             w,h = x2-x1,y2-y1
#
#             # confidence
#             conf = math.ceil((box.conf[0]*100))/100 #it's 2 Decimal Points we use multiple with 100 and divide by 100
#             # print(conf)
#             #cvzone.putTextRect(frame,f'{conf}',(max(0,x1),max(35,y1)))
#
#             #class name
#             cls = int(box.cls[0])
#             CurrentClass = ClassNames[cls]
#             if (CurrentClass in ['car', 'truck', 'bicycle', 'motorbike', 'bus']) and conf > 0.3:
#                 #cvzone.putTextRect(frame,f'{CurrentClass} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1,offset=3)
#                 #cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=5)
#                 currentArray =np.array([x1,y1,x2,y2,conf])
#                 detections = np.vstack((detections,currentArray))
#     resultTracker = tracker.update(detections)
#     cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,0),2)
#     for result in resultTracker:
#         x1,y1,x2,y2,id = result
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         w, h = x2 - x1, y2 - y1
#
#         cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2,colorR=(255,0,255))
#         cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
#
#
#         cx,cy = x1+w//2,y1+h//2
#         cv2.circle(frame,(cx,cy),3,(255,0,255),cv2.FILLED)
#
#         if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 :
#             if totalcount.count(id) ==0:
#                 totalcount .append(id)
#                 cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
#
#     cv2.putText(frame,f'Out: {len(totalcount)}',(25,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
#
#     cv2.imshow('Video_Capture',frame)
#     cv2.imshow('Masking',masked_frame)
#
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#





