import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def PerspectiveTrans(img):
    src = np.float32([[490, 482], [810, 482],
                      [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])

    M =cv2.getPerspectiveTransform(src,dst)
    if M is not None:
        warped = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    else:
        print("The M is none")
    return warped

def test():
    img = plt.imread("../test_images/test1.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    warped = PerspectiveTrans(gray)
    plt.imshow(warped)
    plt.show()

if __name__=="__main__":
    test()
