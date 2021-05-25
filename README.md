# phone-detector-opencv
Detect a phone in a picture of various backgrounds using OpenCV's image processing feature contour detection and filtering based on general geometric properties (area, verticies, arc length, etc.).

Usage instructions:
Call the Python 3 interpreter with an argument to a jpg image in the dataset.

python3 find_phone.py dataset/X.jpg

The output will be normalized (0,1) coordinates of the center of the cell phone, or -1, -1 if no valid contours are found. 
