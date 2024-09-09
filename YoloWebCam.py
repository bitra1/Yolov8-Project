from ultralytics import YOLO
import cv2
import cvzone  #display detection
import math

#cap = cv2.VideoCapture(0)
# cap.set(3,1200)
# cap.set(4,720)

cap = cv2.VideoCapture("../Videos/c1_12.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

ClassNames =["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                        "scissors", "teddy bear", "hair drier", "toothbrush" ]

while True:
    ret,frame = cap.read()
    result = model(frame,stream=True)
    for r in result:
        boxes = r.boxes

        #bounding box
        for box in boxes:
            x1,y1,x2,y2 =box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(frame,(x1,y1,w,h))
            #print('The Box value=',box)

            # confidence
            conf = math.ceil((box.conf[0]*100))/100 #it's 2 Decimal Points we use multiple with 100 and divide by 100
            # print(conf)
            #cvzone.putTextRect(frame,f'{conf}',(max(0,x1),max(35,y1)))

            #class name
            cls = int(box.cls[0])
            cvzone.putTextRect(frame,f'{ClassNames[cls]},{conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)

    cv2.imshow('Video_Capture',frame)
    cv2.waitKey(1)



