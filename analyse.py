import crop
import torch
import os
from PIL import Image
from torchvision import transforms
import argparse

crop_dir = "crops"
model_classify = torch.load(os.path.join("models", "classify images","best.pt"), map_location=torch.device('cpu'))

# take image name as argument
parser = argparse.ArgumentParser()
parser.add_argument("image_name", help="Name of the image to be analysed")
parser.add_argument("--crop_dir", help="Name of the directory to save cropped images")
args = parser.parse_args()
image_name = args.image_name
if args.crop_dir is not None:
    crop_dir = args.crop_dir

crop.crop(img_name=image_name, crop_dir_name=crop_dir)

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

for img_name in os.listdir(crop_dir):
    im0 = Image.open(os.path.join(crop_dir, img_name))
    im0 = transform(im0)
    output = model_classify(im0.unsqueeze(0))
    pred = torch.round(output)
    class_count[pred.item()] += 1

print("Number of images of each class:")
print("Himalayan Products: ", class_count[1])
print("Non-Himalayan Products: ", class_count[0])
print(f"{(class_count[1]/(class_count[1]+class_count[0])*100)}% of the images are himalayan products")

    

