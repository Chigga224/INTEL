#import all the necessary libraries
import os
import cv2
import torch
from ultralytics import YOLO

#initialize the model and provide the video path
#we provide several example footages and all the footages were collected in real time
model = YOLO('yolov8m.pt')
video_path = 'testtt.mp4'
cap = cv2.VideoCapture(video_path)

#resize the frames into the provided video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#create a working window
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', frame_width, frame_height)

#initialize the threshold values
alert_threshold = 0.7 
fps = cap.get(cv2.CAP_PROP_FPS)
alert_frame_threshold = int(alert_threshold * fps)

cut_in_detected = False
cut_in_start_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    car_detected = False
#define several types of vehicles from the given weight "yolov8m.pt"
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if model.names[cls] == 'car':
                car_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Car {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif model.names[cls] == 'truck':
                truck_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Truck {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
            elif model.names[cls] == 'bus':
                bus_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Bus {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
            elif model.names[cls] == 'motorcycle':
                bike_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, f'Bike {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)         
  #introduced os library to take a snap of vehicle cut-in detected frame and an if statement is used as condition of detection          
    if car_detected or truck_detected or bus_detected or bike_detected:
        if not cut_in_detected:
            cut_in_detected = True
            cut_in_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            #The frame will be provided with a string that says cut in detected
            cv2.putText(frame, 'Vehicle cut-in detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            save_dir = 'pythonProject1\\Test' 
            os.makedirs(save_dir, exist_ok=True)
    else:
        cut_in_detected = False
#This prints the output as cut in detected if the cut in frame extends for 0.7 seconds
    if cut_in_detected:
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_frame - cut_in_start_frame >= alert_frame_threshold:
            print("ALERT: Vehicle cut-in detected for 0.7 seconds!")
            cut_in_detected = False 
#A save path is assigned for the output to store
            save_path = os.path.join(save_dir, f'cut_in_frame_{int(current_frame)}.jpg')
            cv2.imwrite(save_path, frame) 
#To show the created workspace window            
    cv2.imshow('Frame', frame)
  #Press q to break the loop and hold 1 to stop the analysis frame by frame  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Destroy the created windows
cap.release()
cv2.destroyAllWindows()
