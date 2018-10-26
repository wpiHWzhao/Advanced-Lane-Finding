# The main goes here
import numpy as np
import cv2
from utils.Calibartion import *
from utils.Convert2Binary import *
from utils.PerspectiveTrans import *
from utils.FindLane import *
img=plt.imread("test_images/test1.jpg")


objPoints, imgPoints = Derive_Points_from_board()
#print(imgPoints)
unDist = Undistort(img,objPoints,imgPoints)
Binary = Convert2Binary(unDist)
Warped = PerspectiveTrans(Binary)
window_width=50
window_height = 80
margin = 80
left_lane_x, left_lane_y, right_lane_x, right_lane_y, masked_Warped = draw_lane_pix(Warped, window_width, window_height, margin)
polyfit_image = fit_poly(left_lane_x,left_lane_y,right_lane_x,right_lane_y, masked_Warped)
plt.imshow(polyfit_image)
plt.show()