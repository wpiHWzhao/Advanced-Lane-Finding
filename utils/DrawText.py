## Udacity Project: Advance Lane Finding.
# This code is to print text in the video.
# Developed by Haowei Zhao, Oct, 2018.

import numpy as np
import cv2

def DrawText(img,left_R, right_R,bottom_lane_position):

    xm_per_pix = 3.7 / 700

    lane_mid = np.average(bottom_lane_position)*xm_per_pix
    # Assuming car is approximately at the middle
    car_R = np.average([left_R,right_R])
    # Calculate the offset
    car_pos = img.shape[1]/2*xm_per_pix
    car_offset = car_pos-lane_mid

    if abs(car_R)>2500:# This value is not necessarily right. Only a empirical value
        text1 = 'The lane is straight'
    else:
        text1 = 'Curve Radius : '+'{:04.2f}'.format(abs(car_R))+' m'


    cv2.putText(img,text1,(40,70),cv2.FONT_HERSHEY_DUPLEX,1.5,(200,255,255),2,cv2.LINE_AA)

    if car_offset>0:
        direction_p = 'Right'
    else:
        direction_p = 'Left'

    text2 = '{:04.2f}'.format(abs(car_offset))+' m '+direction_p+' of center'

    cv2.putText(img,text2,(40,120),cv2.FONT_HERSHEY_DUPLEX,1.5,(200,255,255),2,cv2.LINE_AA)

    return img

