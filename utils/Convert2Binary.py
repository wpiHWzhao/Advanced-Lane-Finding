import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#image = plt.imread('../test_images/test1.jpg')



def Convert2Binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # CV_64F is the depth of the image, 1 is the derivative of x, 0 means we do not need the derivitve on y

    abs_sobelx = np.absolute(sobelx)

    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))# Covert to (0-255)


    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1


    # Stack each channel
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary



