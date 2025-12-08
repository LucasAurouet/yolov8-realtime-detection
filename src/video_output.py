import cv2
from ultralytics import YOLO
import torch

def inference(model, frame):

    results = model(frame)    

    return results[0].boxes, results[0].probs

def draw_boxes(frame, boxes, probas):
        for box in boxes:
            x1, y1, x2, y2 = boxes.xyxy
            conf = boxes.conf
            # Class and confidence for the current cell
            class_id = probas.argmax().item()
            # Draw the box
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
            cv2.putText(frame, f"{class_id}:{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        # Show the image 
        cv2.imshow('video', frame)

def read_stream(model=None):

    stream = cv2.VideoCapture(0)

    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    
    while True:
        ret, frame = stream.read()
        fps = stream.get(cv2.CAP_PROP_FPS)
        
        if not ret:
            print('invalid stream')
            break
        
        if model != None:
            boxes, probas = inference(model, frame)
            draw_boxes(frame, boxes, probas)
        else:    
            cv2.imshow('video', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    stream.release()
    cv2.destroyAllWindows()
    print(f'{fps} FPS')
    
model = YOLO('D:\\yolov8-realtime-detection\\weights\\finetuned_weights.pt')
model.to("cuda")
model.half()

read_stream(model=None)