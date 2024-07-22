# the following is a file that crops images using the trained YOLO model
import os
import cv2
from ultralytics import YOLO

def crop(img_name, crop_dir_name="crops", model_path=os.path.join("models", "object detection", "best.pt")):
    model = YOLO(model_path)
    if not os.path.exists(crop_dir_name):
        os.mkdir(crop_dir_name)

    # img_file is only the last part of image name
    img_file = img_name.split(os.sep)[-1]
    im0 = cv2.imread(img_name)
    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    if boxes is not None:
        idx=0

        for box in boxes:

            crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

            cv2.imwrite(os.path.join(crop_dir_name, img_file + str(idx) + ".png"), crop_obj)
            idx+=1
