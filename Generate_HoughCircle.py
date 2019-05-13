# import libraries
import numpy as np
import cv2
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

# Load picture
def Getcircles(I):

    im_path = I
    img = cv2.imread(im_path)

    # Convert image to greyscale and compute edges

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M = canny(grey, sigma=3, low_threshold=10, high_threshold=50)

    # Create Hough Space And Detect Circles Within The Radius Range Of 100-300 given Step size of 1

    hough_radii = np.arange(100,300, 1)
    hough_res = hough_circle(M, hough_radii)

    # Select the most prominent circle from Hough Space

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    return cx[0],cy[0],radii[0]
