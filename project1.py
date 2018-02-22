# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:29:22 2018

see: http://localhost:8888/notebooks/P1.ipynb

@author: michaelg@bluemetal.com
"""

import os
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import pandas as pd
#%matplotlib inline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def GetSlope(x1,y1,x2,y2):
    """
    Note: this function finds the slope of a line. Returns a big number of dx = 0
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx != 0:
        m = dy/dx
    else:
        m = 100000000.0
    return m

def LinesToDf(lines):
    """
    Note: this function converts the lines array to a pandas dataframe
    """
    cols = ['x1','y1','x2','y2','m','b']
    df = pd.DataFrame(columns = cols)
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = round(GetSlope(x1,y1,x2,y2),4)
            b = y1 - m*x1
            df = df.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'m':m,'b':b}, ignore_index=True)
    return df

def GetYmin(l1,l2,xmid):
    """
    Note: this function finds the y at which the two lane lines converge in the distance.
    It's important to adjust ymin for hills.
    """
    ymin1 = l1.iloc[0]['m']*xmid + l1.iloc[0]['b']
    ymin2 = l2.iloc[0]['m']*xmid + l2.iloc[0]['b']
    ymin = (ymin1 + ymin2)/2
    return int(ymin)

def GetLineIntersection(line1, line2):
    """
    Note: this function finds the interesection between two lines. It's used here to
    find where the lane lines converge in the distance.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div[0] == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x[0]), int(y[0])

def draw_lines(img, lines, color=[255, 0, 0], thickness=15):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # put lines array in a dataframe
    df = LinesToDf(lines)
    # find pos and neg sloped lines
    negM = df[df['m'] < 0]['m']
    posM = df[df['m'] > 0]['m']
    # find median slope on left and right (statistics.median was giving me strange answers)
    # TODO: this might lead to problems if masking isn't very good
    negM = sorted(negM)
    posM = sorted(posM)
    negMedian = negM[int(len(negM)/2)]
    posMedian = posM[int(len(posM)/2)]
    # find the lines on the left and right with the median +/- slope
    l1 = df[df['m']==posMedian]
    l2 = df[df['m']==negMedian]
    l1 = l1.reset_index(drop=True)
    l2 = l2.reset_index(drop=True)
    
    imshape = img.shape

    # xmid is the approximate intersection of the left and right lines in the distance
    xmid, ymid = GetLineIntersection(([l1['x1'], l1['y1']],[l1['x2'], l1['y2']]), 
                                     ([l2['x1'], l2['y1']],[l2['x2'], l2['y2']]))

    # ymin is the upper-most part of the masked area - the tip of the triangle
    ymin = GetYmin(l1,l2,xmid)

    # find x & y for each line at the bottom of the image (ymax)    
    xrt = int((imshape[0] - l1.iloc[0]['b'])/l1.iloc[0]['m'])
    xrt = min(xrt,0.9*imshape[1])
    yrt = imshape[0]    
    xlt = int((imshape[0] - l2.iloc[0]['b'])/l2.iloc[0]['m'])
    xlt = max(xlt,0.1*imshape[1])
    ylt = imshape[0]
    
    cols = ['x1','y1','x2','y2']
    lines = pd.DataFrame(columns=cols)
    lines = lines.append({'x1':xrt,'y1':yrt,'x2':xmid+5,'y2':ymin}, ignore_index=True)
    lines = lines.append({'x1':xlt,'y1':ylt,'x2':xmid-5,'y2':ymin}, ignore_index=True)
    for i,line in lines.iterrows():
        x1 = int(line['x1'])
        y1 = int(line['y1'])
        x2 = int(line['x2'])
        y2 = int(line['y2'])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    """
    to use
    cv2.inRange() for color selection
    cv2.fillPoly() for regions selection
    cv2.line() to draw lines on an image given endpoints
    cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
    cv2.bitwise_and() to apply a mask to an image
    """

    kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    #line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    
    grayImg = grayscale(image)
    blurImg = gaussian_blur(grayImg, kernel_size)
    cannyImg = canny(blurImg, low_threshold, high_threshold)
    
    # orig
    imshape = image.shape
    ymin = 280
    xmid = 485
    xlt = 150
    xrt = imshape[1]-60
    
    # below is my attempt at generalizing the masked region - needs more work
    #ymin = int(0.5 * imshape[0])
    #xmid = int(imshape[1]/2)
    #xlt = int(0.1*imshape[1])
    #xrt = int(0.9*imshape[1]) 
    #xlt = 0
    #xrt = imshape[1]
    
    vertices = np.array([[(xlt,imshape[0]),(xmid-5, ymin), (imshape[1]-xmid+5, ymin), (xrt,imshape[0])]], dtype=np.int32)
    maskedImg = region_of_interest(cannyImg, vertices)
    plt.imshow(maskedImg) 
    
    houghImg = hough_lines(maskedImg, rho, theta, threshold, min_line_len, max_line_gap)
    plt.imshow(houghImg) 
    
    α=0.8
    β=1.
    γ=0.
    wtImg = weighted_img(houghImg, image, α, β, γ)
    plt.imshow(wtImg) 

    return wtImg

"""
#reading in an image
image = mpimg.imread(wd+'test_images\\solidWhiteRight.jpg')
process_image(image)

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
"""


images = os.listdir("test_images/")
images_output = 'test_images_output'
# using the pipeline to process images
for imFile in images:
    image = mpimg.imread('test_images\\'+imFile)
    img = process_image(image)
    cv2.imwrite(images_output + "\\" + imFile, img)


white_output ='test_videos_output\\solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

#clip1 = VideoFileClip(wd+"test_videos\\solidWhiteRight.mp4")
clip1 = VideoFileClip("test_videos\\solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
clip1.reader.close()

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)
clip2.reader.close()

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
clip3.reader.close()


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

