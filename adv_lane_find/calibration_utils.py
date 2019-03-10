import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os.path as path
import pickle

def lazy_calibration(func):
    """ Detection Whether Camera is calibrated or not"""
    calibration_cache='camera_cal/calibration_data.pickle'

    def wrapper(*args,**kwargs):
        if path.exists(calibration_cache):
            print('Loading cached camera calibration .. ', end =' ' )
            with open(calibration_cache,'rb') as dump_file:
                calibration = pickle.load(dump_file)
        else:
            print('Computing camera calibration',end=' ')
            calibration = func(*args,**kwargs)
            with open(calibration_cache,'wb') as dump_file:
                pickle.dump(calibration,dump_file)
        print('Done.')
        return calibration
    return wrapper

@lazy_calibration

def calibrate_camera(calib_images_dir,verbose=False):
    """
    Calibrate the camera given a directory containing calinration chessboards.
    """
    assert path.exists(calib_images_dir),'"{}" must exist and contain calibration images.'.format(calib_images_dir)

    # Preparing object points
    objp=np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Selecting calibration images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Making list of calibration images
    images  = glob.glob(calib_images_dir+'/*.jpg')
    # images  = glob.glob(path.join(calib_images_dir,'calibration*.jpg'))
    # print(images)
    # Step for the list and search for chessboard corners
    for filename in images:
        img=cv2.imread(filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        pattern_found,corners = cv2.findChessboardCorners(gray,(9,6),None)

        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if verbose:
                img = cv2.drawChessboardCorners(img,(9,6),corners,pattern_found)
                cv2.imshow('img',img)
                cv2.waitkey(500)
    if verbose:
        cv2.destroyAllWindows()
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

    return ret,mtx,dist,rvecs,tvecs

def undistort(frame,mtx,dist,verbose=False):
    """Undistoring the frame given the camera matrix and distortion coefficient """

    frame_undistorted = cv2.undistort(frame,mtx,dist,newCameraMatrix = mtx)
    if verbose:
        fig,ax = plt.subplots(nrows=1,ncols=2)
        ax[0].imshow(cv2.cvtColor(frame,cv2.Color_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(frame_undistorted,cv2.Color_BGR2RGB))
        plt.show()

    return frame_undistorted

# Vanishing Point calculation has to be taken out

if __name__ == "__main__":
    # ret,mtx,dist,rvecs,tvecs = calibrate_camera(calib_images_dir='../data/calibration/camera_center')
    ret,mtx,dist,rvecs,tvecs = calibrate_camera(calib_images_dir='camera_cal')
    print(mtx)
    img = cv2.imread('test_images/test2.jpg')
    img_undistorted = undistort(img,mtx,dist)
    # cv2.imshow('undistorted',img_undistorted)
    # cv2.waitKey(2000)
    # cv2.imwrite('../Results/Camera_calibration/test_calibration_before.jpg',img)
    # cv2.imwrite('../Results/Camera_calibration/test_calibration_after.jpg',img_undistorted)