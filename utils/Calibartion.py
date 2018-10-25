import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mping

def Derive_Points_from_board():
    images = glob.glob("camera_cal/calibration*.jpg")

    #images = glob.glob("../camera_cal/calibration*.jpg") # This line is for test mode.Uncomment it if you need

    # Setup arrays to hold points
    objPoints = []
    imgPoints = []

    # Prepare objPoints
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgPoints.append(corners)
            objPoints.append(objp)

            #img_draw = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            #cv2.imshow("1",img_draw)
            #cv2.waitKey(0)
    #print(objPoints)
    return objPoints, imgPoints

def Undistort(img,objPoints,imgPoints):

    if objPoints == None or imgPoints == None:
        print("The input martix are None")
        return None
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints,imgPoints,gray.shape[::-1],None,None)

    if ret:
        dst = cv2.undistort(img,mtx,dist,None,mtx)
    else:
        print('Calibration Failed')

    return dst
def test():
    image = cv2.imread("../camera_cal/calibration2.jpg")

    objPoints, imgPoints = Derive_Points_from_board()
    undist = Undistort(image,objPoints, imgPoints)
    plt.imshow(undist)
    plt.show()

if __name__=="__main__":
    print("Test mode")
    test()



