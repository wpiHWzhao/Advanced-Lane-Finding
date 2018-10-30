## Udacity Project: Advance Lane Finding.
# This code is to convert image to binary .
# Developed by Haowei Zhao, Oct, 2018.


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#image = plt.imread('../test_images/test1.jpg')



def Convert2Binary_Sobel_S(img, s_thresh=(125, 255), sx_thresh=(20, 100)):

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    blur_hls = cv2.GaussianBlur(hls,(11,11),0)
    l_channel = blur_hls[:, :, 1]
    s_channel = blur_hls[:, :, 2]
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

# Convert to LAB color space
def Convert2Binary_LAB_L(img, lab_thresh = (190,255),l_thresh=(220,255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_b = lab[:, :, 2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b * (255 / np.max(lab_b))
    # Apply a threshold to the B channel
    output_lab = np.zeros_like(lab_b)
    output_lab[((lab_b > lab_thresh[0]) & (lab_b <= lab_thresh[1]))] = 1

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # Apply a threshold to the L channel
    output_l = np.zeros_like(hls_l)
    output_l[(hls_l > l_thresh[0]) & (hls_l <= l_thresh[1])] = 1

    binary_output = np.zeros_like(output_lab)
    binary_output[(output_lab==1)|(output_l==1)]=1

    return binary_output



