import numpy as np
import cv2

def DrawText(img,car_R,car_offset):

    if car_R>0:
        direction_r = 'Left'
    else:
        direction_r = 'Right'

    if abs(car_R)>5000:
        text1 = 'The lane is straight'
    else:
        text1 = 'Curve Radius : '+direction_r+' '+'{:04.2f}'.format(abs(car_R))+' m'
    cv2.putText(img,text1,(40,70),cv2.FONT_HERSHEY_DUPLEX,1.5,(200,255,255),2,cv2.LINE_AA)
    if car_offset>0:
        direction_p = 'Right'
    else:
        direction_p = 'Left'

    text2 = '{:04.2f}'.format(abs(car_offset))+' m '+direction_p+' of center'

    cv2.putText(img,text2,(40,120),cv2.FONT_HERSHEY_DUPLEX,1.5,(200,255,255),2,cv2.LINE_AA)

    return img

