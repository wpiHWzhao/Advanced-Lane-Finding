import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.PerspectiveTrans import *
import matplotlib.image as mpimg

def find_window(image,window_width,window_height,margin):
    window_center = []
    bottom_lane_position = []

    window = np.ones(window_width)
    offset = window_width / 2

    # Start with the first slice on y direction
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)],axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-offset

    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):],axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-offset + int(image.shape[1]/2)
    window_center.append((l_center,r_center))
    bottom_lane_position.append((l_center,r_center))



    for slice_y_ind in range(1,int(image.shape[0]/window_height)):
        image_y_slice = np.sum(image[int(image.shape[0]-(slice_y_ind+1)*window_height):int(image.shape[0]-slice_y_ind*window_height),:],axis=0)
        convolve = np.convolve(window,image_y_slice)
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(convolve[l_min_index:l_max_index])-offset+l_min_index


        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(convolve[r_min_index:r_max_index]) - offset + r_min_index
        window_center.append((l_center,r_center))

    return window_center,bottom_lane_position

def make_mask(window_width,window_height,img,centers,slice):
    mask = np.zeros_like(img)
    mask[int(img.shape[0]-(slice+1)*window_height):int(img.shape[0]-slice*window_height),max(int(centers-window_width/2),0):min(int(centers+window_width/2),img.shape[1])] = 1
    return mask

def draw_lane_pix(image,window_width,window_height,margin):# Input image need to be a warped one
    window_center,bottom_lane_position = find_window(image,window_width,window_height,margin)
    left_lane_x =[]
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []
    if window_center is not None:
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        for slice_y_ind in range(0,int(image.shape[0]/window_height)):
            l_mask = make_mask(window_width,window_height,image,window_center[slice_y_ind][0],slice_y_ind)
            r_mask = make_mask(window_width,window_height,image,window_center[slice_y_ind][1],slice_y_ind)

            l_points[(l_mask == 1)] = 255
            nonzero_l = l_points.nonzero()
            left_lane_x.append(nonzero_l[1])
            left_lane_y.append(nonzero_l[0])
            r_points[(r_mask == 1)] = 255
            nonzero_r = r_points.nonzero()
            right_lane_x.append(nonzero_r[1])
            right_lane_y.append(nonzero_r[0])

        left_lane_x = np.concatenate(left_lane_x)
        left_lane_y = np.concatenate(left_lane_y)
        right_lane_x = np.concatenate(right_lane_x)
        right_lane_y = np.concatenate(right_lane_y)

        image_position_of_windows_channel = np.array(l_points+r_points,np.uint8)
        zero_channel = np.zeros_like(image_position_of_windows_channel)
        image_position_of_windows = np.array(cv2.merge((zero_channel,image_position_of_windows_channel,zero_channel)),np.uint8)
        color_img = np.dstack((image,image,image))*255
        masked_img = cv2.addWeighted(color_img,1,image_position_of_windows,0.5,0)
    else:
        print("The window_center is not found")
        return None

    return left_lane_x,left_lane_y,right_lane_x,right_lane_y,masked_img,bottom_lane_position
    # return left_lane_x, left_lane_y, right_lane_x, right_lane_y, bottom_lane_position


def fit_poly(left_lane_x,left_lane_y,right_lane_x,right_lane_y,masked_img):

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    left_fit = np.polyfit(left_lane_y,left_lane_x,2)
    right_fit = np.polyfit(right_lane_y,right_lane_x,2)

    left_fit_real = np.polyfit(left_lane_y*ym_per_pix, left_lane_x*xm_per_pix, 2)
    right_fit_real = np.polyfit(right_lane_y*ym_per_pix, right_lane_x*xm_per_pix, 2)

    ploty = np.linspace(0,masked_img.shape[0]-1,masked_img.shape[0])

    # Fit left and right lane with second order polynomial
    try:
        left_poly_x = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        right_poly_x = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    except TypeError:
        print("The function can not fit a line")
        left_poly_x = ploty**2+ploty
        right_poly_x = ploty**2+ploty

    plt.plot(left_poly_x,ploty,color = 'red')
    plt.plot(right_poly_x,ploty,color = 'blue')

    y_eval = np.max(ploty) * ym_per_pix

    left_R = ((1 + (2 * left_fit_real[0] * y_eval + left_fit_real[1]) ** 2) ** (3 / 2)) / np.absolute(2 * left_fit_real[0])
    right_R = ((1 + (2 * right_fit_real[0] * y_eval + right_fit_real[1]) ** 2) ** (3 / 2)) / np.absolute(2 * right_fit_real[0])

    # lane_mid = np.average(bottom_lane_position)*xm_per_pix
    # print(left_R,right_R)
    # if left_R > right_R:
    #     car_R = -np.average([left_R,right_R])
    # else:
    #     car_R = np.average([left_R,right_R])
    #
    # car_pos = masked_img.shape[1]/2*xm_per_pix
    # car_offset = car_pos-lane_mid

    return masked_img,left_fit,right_fit,left_R,right_R,ploty

def unwarp_with_lane(warped,left_fit,right_fit,ploty):


    left_poly_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_poly_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    warped_blank = np.zeros_like(warped).astype(np.uint8)

    color_warped = np.dstack((warped_blank,warped_blank,warped_blank))
    pts_left = np.array([np.transpose(np.vstack([left_poly_x,ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_poly_x,ploty])))])
    pts = np.hstack((pts_left,pts_right))

    cv2.fillPoly(color_warped,np.int_([pts]),(0,255,0))

    cv2.polylines(color_warped,np.int_([pts_left]), isClosed = False,color=(255,0,0),thickness=20)
    cv2.polylines(color_warped, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)

    unwarped = InversePerspectiveTrans(color_warped)

    return unwarped


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    masked_img, left_fit, right_fit, left_R, right_R, ploty = fit_poly(leftx, lefty, rightx, righty, binary_warped)

    # ## Visualization ##
    # # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # window_img = np.zeros_like(out_img)
    # # Color in left and right line pixels
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #
    # # Generate a polygon to illustrate the search window area
    # # And recast the x and y points into usable format for cv2.fillPoly()
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
    #                                                                 ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
    #                                                                  ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))
    #
    # # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #
    # # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # ## End visualization steps ##

    return masked_img, left_fit, right_fit, left_R, right_R, ploty


def test():
    img = plt.imread("warped_example.jpg")
    window_width = 50
    window_height = 80
    margin = 80
    left_lane_x, left_lane_y, right_lane_x, right_lane_y, masked_img = draw_lane_pix(img, window_width, window_height, margin)
    polyfit_image,car_R,car_offset,ploty= fit_poly(left_lane_x,left_lane_y,right_lane_x,right_lane_y,masked_img)
    print(car_R,car_offset)
    plt.imshow(polyfit_image)
    plt.show()


if __name__ == "__main__":
    test()
