# Shelf Analytics
## Objective:
- analysis of shelves in retail stores from image data
- the main point of analysis was number of products of a particular brand visible on the shelf from the image

## Our pipeline:
- run YOLO on the image and extract data on locations and bounding boxes for products
- crop out the objects using results from YOLO
- classify the cropped images based on whether they are of the brand being checked or not using a simple CNN with pretrained weights from ResNet
- some images will be ambiguously classified and for these, use their proximity data, i.e. check their surroundings for well classified images and use those results to conclude the class of the ambiguous images (not yet integrated into the pipeline)
- color clustering is a technique that can fit into into the pipeline in the future

## Usage:
- start by cloning the repository as follows

        git clone https://github.com/s0han24/shelf_analytics.git

- enter the repository directory

        cd shelf_analytics

- install the project requirements(assuming python3 pip is installed on the computer)

        python3 -m pip install -r requirements.txt

- to run analysis on an image of a shelf

        python3 analyse.py "<path to image>"

- to get information about other arguments

        python3 analyse.py --help

- to use draw_box_static.py for visual analysis of the image

        python3 draw_box_static.py "<path to image>"

## Additional information
- draw_box_static.py takes an image of a shelf as input and gives out an annotated image with boxes drawn around each product as output

- proximity.ipynb demonstrates how we can take proximity information into account for classification.

- color-clustering.ipynb shows a way to identify similar products based on color scheme

- object-detection.ipynb shows how to fine tune a YOLO model to detect objects on a shelf