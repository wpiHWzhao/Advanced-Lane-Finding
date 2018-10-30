import numpy as np

def fit_poly_real(left_lane_x,left_lane_y,right_lane_x,right_lane_y,masked_img):

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    left_fit_real = np.polyfit(left_lane_y*ym_per_pix, left_lane_x*xm_per_pix, 2)
    right_fit_real = np.polyfit(right_lane_y*ym_per_pix, right_lane_x*xm_per_pix, 2)

    return masked_img,left_fit_real,right_fit_real

def CalculateRadius(left_fit_real,right_fit_real,image):

    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    y_eval = np.max(ploty) * ym_per_pix

    left_R = ((1 + (2 * left_fit_real[0] * y_eval + left_fit_real[1]) ** 2) ** (3 / 2)) / np.absolute(
        2 * left_fit_real[0])
    right_R = ((1 + (2 * right_fit_real[0] * y_eval + right_fit_real[1]) ** 2) ** (3 / 2)) / np.absolute(
        2 * right_fit_real[0])

    return left_R,right_R
