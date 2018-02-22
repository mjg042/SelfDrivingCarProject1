# Self Driving Car Project 1 - Finding Lane Lines on the Road
This doc describes the work I did for the Udacity Self Driving Car Finding Lane Lines on the Road Project.

You can either download the project1.py file or load the P1.ipynb Jupyter notebook to duplicate the work I did.

The original project description can be found here. https://github.com/udacity/CarND-LaneLines-P1

The basic challenge of this project is to take images like this:

[//]: # (Image References)

[image1]: ./test_images/solidWhiteRight.jpg "Original"
[image2]: ./examples/laneLines_thirdPass.jpg "Original with lane lines"


![Original image][image1]

and paint lines on the image like this:

![Original image with lane lines][image2]

Moreover, the project solution had to process a video taken from cars driving on the highway.

To accomplish this task, I built an image processing pipeline. 

Read image
Convert image to grayscale
Define kernel size and apply Gaussian smoothing for slight blurring
Define parameters for Canny transformation
Canny transform to find edges in image
Mask image using cv2.fillPoly() to remove everything outside region of interest
Define Hough transform parameters and run Hough transform on masked edge-detected image
Draw line segments
Draw lines extrapolated from line segments
Overlay lines on original image
