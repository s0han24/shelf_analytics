import crop
import torch
import os
import cv2
from torchvision import transforms

model_classify = torch.load("models\\classify images\\best.pt")

crop.crop()

# run the model on the cropped images and count number of images of each class
class_count = {0: 0, 1: 0}

transform = transforms.Compose([        # Defining a variable transforms
 transforms.Resize(256),                # Resize the image to 256×256 pixels
 transforms.CenterCrop(224),            # Crop the image to 224×224 pixels about the center
 transforms.ToTensor(),                 # Convert the image to PyTorch Tensor data type
 transforms.Normalize(                  # Normalize the image
 mean=[0.485, 0.456, 0.406],            # Mean and std of image as also used when training the network
 std=[0.229, 0.224, 0.225]      
)])

for img_name in os.listdir("crops"):
    im0 = cv2.imread(os.path.join("crops", img_name))
    im0 = transform(im0)
    pred = torch.round(model_classify(im0))
    class_count[pred.item()] += 1

print(class_count)

    

