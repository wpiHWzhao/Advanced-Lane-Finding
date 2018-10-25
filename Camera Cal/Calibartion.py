import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping

img = cv2.imread("calibration1.jpg")
if img is not None:
    cv2.imshow("Readin.jpg",img)
else:
    print("no")
