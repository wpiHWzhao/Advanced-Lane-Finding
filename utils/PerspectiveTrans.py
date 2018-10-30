import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def PerspectiveTrans(img):
    src = np.float32([[490, 480], [810, 480],
                      [1250, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])


    M =cv2.getPerspectiveTransform(src,dst)
    if M is not None:
        warped = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    else:
        print("The M is none")
    return warped

def InversePerspectiveTrans(img):
    dst = np.float32([[490, 480], [810, 480],
                      [1250, 720], [40, 720]])
    src = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])


    Minv = cv2.getPerspectiveTransform(src, dst)
    if Minv is not None:
        unwarped = cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]))
    else:
        print("The M is none")
        return None
    return unwarped

def test():
    img = plt.imread("../test_images/test1.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    warped = PerspectiveTrans(gray)
    plt.imshow(warped)
    plt.show()

if __name__=="__main__":
    test()
