import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_window(image,window_width,window_height,margin):
    window_center = []
    window = np.ones(window_width)
    offset = window_width / 2
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)],axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-offset
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):],axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-offset + int(image.shape[1]/2)
    window_center.append((l_center,r_center))

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

    return window_center

def make_mask(window_width,window_height,img,centers,slice):
    mask = np.zeros_like(img)
    mask[int(img.shape[0]-(slice+1)*window_height):int(img.shape[0]-slice*window_height),max(int(centers-window_width/2),0):min(int(centers+window_width/2),img.shape[1])] = 1
    return mask

def draw_lane_pix(image,window_width,window_height,margin):# Input image need to be a warped one
    window_center = find_window(image,window_width,window_height,margin)
    if window_center is not None:
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        for slice_y_ind in range(0,int(image.shape[0]/window_height)):
            l_mask = make_mask(window_width,window_height,image,window_center[slice_y_ind][0],slice_y_ind)
            r_mask = make_mask(window_width,window_height,image,window_center[slice_y_ind][1],slice_y_ind)

            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        image_position_of_windows_channel = np.array(l_points+r_points,np.uint8)
        zero_channel = np.zeros_like(image_position_of_windows_channel)
        image_position_of_windows = np.array(cv2.merge((zero_channel,image_position_of_windows_channel,zero_channel)),np.uint8)
        color_img = np.dstack((image,image,image))*255
        masked_img = cv2.addWeighted(color_img,1,image_position_of_windows,0.5,0)
    else:
        print("The window_center is not found")
        return None

    return masked_img

def test():
    img = plt.imread("warped_example.jpg")
    window_width = 50
    window_height = 80
    margin = 80
    masked_Warped = draw_lane_pix(img, window_width, window_height, margin)
    plt.imshow(masked_Warped)
    plt.show()


if __name__ == "__main__":
    test()
