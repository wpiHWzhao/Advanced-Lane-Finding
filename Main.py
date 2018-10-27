# The main goes here
import numpy as np
import cv2
from utils.Calibartion import *
from utils.Convert2Binary import *
from utils.PerspectiveTrans import *
from utils.FindLane import *
from utils.DrawText import *
from moviepy.editor import VideoFileClip

# str_l1 = plt.imread("test_images/straight_lines1.jpg")
# str_l2 = plt.imread("test_images/straight_lines2.jpg")
# test1 = plt.imread('test_images/test1.jpg')
# test2 = plt.imread('test_images/test2.jpg')
# test3 = plt.imread('test_images/test3.jpg')
# test4 = plt.imread('test_images/test4.jpg')
# test5 = plt.imread('test_images/test5.jpg')
# test6 = plt.imread('test_images/test6.jpg')
class AdvanceLaneDetect:
    def __init__(self):
        self.objPoints, self.imgPoints = Derive_Points_from_board()

    def pipline(self,img):
        unDist = Undistort(img, self.objPoints, self.imgPoints)
        Binary = Convert2Binary(unDist)
        Warped = PerspectiveTrans(Binary)

        window_width = 50
        window_height = 80
        margin = 80
        left_lane_x, left_lane_y, right_lane_x, right_lane_y, masked_Warped, bottom_lane_position = draw_lane_pix(Warped,
                                                                                                                  window_width,
                                                                                                                  window_height,
                                                                                                                  margin)

        polyfit_image, car_R, car_offset, left_poly_x, right_poly_x, ploty = fit_poly(left_lane_x, left_lane_y,
                                                                                      right_lane_x, right_lane_y,
                                                                                      masked_Warped, bottom_lane_position)

        Unwarped_with_lane = unwarp_with_lane(Warped, left_poly_x, right_poly_x, ploty)
        result = cv2.addWeighted(unDist, 1, Unwarped_with_lane, 0.3, 0)
        result = DrawText(result, car_R, car_offset)

        return result

# Final_str_l1 = pipline(str_l1)
# plt.imsave('output_images/straight_lines1.png',Final_str_l1)
# Final_str_l2 = pipline(str_l2)
# plt.imsave('output_images/straight_lines2.png',Final_str_l2)
# Final_test1 = pipline(test1)
# plt.imsave('output_images/test1.png',Final_test1)
# Final_test2 = pipline(test2)
# plt.imsave('output_images/test2.png',Final_test2)
# Final_test3 = pipline(test3)
# plt.imsave('output_images/test3.png',Final_test3)
# Final_test4 = pipline(test4)
# plt.imsave('output_images/test4.png',Final_test4)
# Final_test5 = pipline(test5)
# plt.imsave('output_images/test5.png',Final_test5)
# Final_test6 = pipline(test6)
# plt.imsave('output_images/test6.png',Final_test6)

Detect = AdvanceLaneDetect()
white_output = 'video_output/project_video.mp4'
clip1 = VideoFileClip('project_video.mp4').subclip(0,5)
white_clip = clip1.fl_image(Detect.pipline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)