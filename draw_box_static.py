import cv2
from ultralytics import YOLO
import os
import argparse

model = YOLO(os.path.join("models", "object detection", "best.pt"))

def draw_bounding_boxes(image, results):
    
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    print(len(results))
    # Iterate through the results
    for box, cls, conf in zip(boxes, classes, confidences):
        x_min, y_min, x_max, y_max = box
        print(box)
        confidence = conf
        name = names[int(cls)]
        
        
        # Draw rectangle on the image
        image = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # Optionally, put class_id text
        image = cv2.putText(image, str(name) + ' ' + str(confidence),(int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # image = cv2.flip(image, 1)
    # Display the image
    cv2.imshow('frame', image)

    

while True:
    args = argparse.ArgumentParser()
    args.add_argument("image", help="Path to the image to be analysed")
    if args.parse_args().image is not None:
        image = args.parse_args().image
    frame = cv2.imread(image)
    results = model([frame])
    draw_bounding_boxes(frame, results)
    if cv2.waitKey(1) == ord('q'):
        break





