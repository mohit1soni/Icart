import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

yellow_HSV_threshold_min=np.array([0,70,70])
yellow_HSV_threshold_max=np.array([50,255,255])


def threshold_frame_in_HSV(frame,min_values,max_values,verbose=False):
    "For the Generation of the threshold of the color frame in HSV space"
    HSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    min_threshold_ok=np.all(HSV > min_values,axis=2)
    max_threshold_ok=np.all(HSV < max_values,axis=2)

    out= np.logical_and(min_threshold_ok,max_threshold_ok)

    if verbose:
        plt.imshow(out,cmap='gray')
        plt.show()
    return out


def threshold_frame_sobel(frame,karnel_size):
    """ Generates Soble Edge Detector Output """
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=karnel_size)
    sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=karnel_size)

    sobel_magnitude=np.sqrt(sobel_x**2+sobel_y**2)
    sobel_magnitude=np.uint8(sobel_magnitude/np.max(sobel_magnitude) * 255)

    _,sobel_magnitude= cv2.threshold(sobel_magnitude,50,1,cv2.THRESH_BINARY)

    return sobel_magnitude.astype(bool)


def get_binary_from_equalized_gray_scale(frame):
    """ To apply histogram equalization to the input frame """
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eq_global  = cv2.equalizeHist(gray)
    _,th=cv2.threshold(eq_global,thresh=250,maxval=255,type=cv2.THRESH_BINARY)
    return th

def binarize(img, verbose=False):
    """ Convert an input image frame to binary frame which highlights the lane lines"""
    h,w=img.shape[:2]
    binary = np.zeros(shape=(h,w),dtype = np.uint8)
    # Making the HSV_Yellow_Mask
    HSV_yellow_mask=threshold_frame_in_HSV(img,yellow_HSV_threshold_min,yellow_HSV_threshold_max,verbose=False)
    binary = np.logical_or(binary,HSV_yellow_mask)

    eq_white_mask=get_binary_from_equalized_gray_scale(img)
    binary = np.logical_or(binary,eq_white_mask)

    # For getting the sobel mask
    sobel_mask = threshold_frame_sobel(img,karnel_size = 9)
    binary=np.logical_or(binary,sobel_mask)

    #  Applying the light Mprphological operation to fill the gaps in the binary images

    karnel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8),cv2.MORPH_CLOSE,karnel)

    if verbose:
        f,ax=plt.subplots(2,3)
        f.set_facecolor('white')
        ax[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        ax[0,0].set_title('Input_Frame')
        ax[0,0].set_axis_off()
        # ax[0,0].set_axis_backgroundcolor('red')

        ax[0,1].imshow(eq_white_mask,cmap='gray')
        ax[0,1].set_title("White Mask")
        ax[0,1].set_axis_off()

        ax[0,2].imshow(HSV_yellow_mask,cmap='gray')
        ax[0,2].set_title("Yellow Mask")
        ax[0,2].set_axis_off()

        ax[1,0].imshow(sobel_mask,cmap='gray')
        ax[1,0].set_title("Sobel Mask")
        ax[1,0].set_axis_off()

        ax[1,1].imshow(binary,cmap='gray')
        ax[1,1].set_title("Before Clousure ")
        ax[1,1].set_axis_off()

        ax[1,2].imshow(closing,cmap='gray')
        ax[1,2].set_title("After Closure")
        ax[1,2].set_axis_off()
        plt.show()

    return closing

if __name__ == "__main__":
    test_images= glob.glob('../data/test_images/*.jpg')
    for test_image in test_images:
        img = cv2.imread(test_image)
        binarize(img=img,verbose=True)

    # cap = cv2.VideoCapture('../data/test_videos/test2.mp4')
    # while True:
    #     ret,frame = cap.read()
    #     binarize(img=frame,verbose=True)
    #     # new_frame =cv2.cvtColor(n_frame,cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow('input_video',frame)
    #     # cv2.imshow('binary_video',n_frame)

    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()