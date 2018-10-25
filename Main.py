# The main goes here
import numpy as np
import cv2
from utils.Calibartion import *
from utils.Convert2Binary import *
from utils.PerspectiveTrans import *

img=plt.imread("test_images/test1.jpg")


objPoints, imgPoints = Derive_Points_from_board()
#print(imgPoints)
unDist = Undistort(img,objPoints,imgPoints)
Binary = Convert2Binary(unDist)
Warped = PerspectiveTrans(Binary)
plt.imshow(Warped)
plt.show()