# shelf_analytics
## Objective:
- analysis of shelves in retail stores from image data
- the main point of analysis was number of products of a particular brand visible on the shelf from the image

## Our pipeline:
- run YOLO on the image and extract data on locations and bounding boxes for products
- crop out the objects using results from YOLO
- classify the cropped images based on whether they are of the brand being checked or not using a simple CNN with pretrained weights from ResNet
- some images will be ambiguously classified and for these, use their proximity data, i.e. check their surroundings for well classified images and use those results to conclude the class of the ambiguous images
- color clustering is a technique that can fit into into the pipeline in the future
