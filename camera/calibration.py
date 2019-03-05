import cv2
import numpy as np
import glob

def capture_single_image():
    cam=cv2.VideoCapture(1)
    cv2.namedWindow('Calibration')
    img_counter = 0

    while True:
        ret,frame = cam.read()
        cv2.imshow('Calibration',frame)
        if not ret:
            break
        k=cv2.waitKey(1)

        if k%256 == 27:
            print("Escaped Hit closing ..")
            break
        elif k%256 == 32:
            img_name="../data/calibration/camera_center/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name,frame)
            print("{} written!".format(img_name))
            img_counter +=1
    cam.release()
    cv2.destroyAllWindows()

def main():
    capture_single_image()


if __name__ == "__main__":
    main()





# # Termination criteria defined
# criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

# # Preparing Object points
# object_points = np.zeros((7*6,3),np.float32)
# object_points[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# # Arrary to store object points
# objpoints=[]
# imgpoints=[]


# images=glob.glob('..\\data\\calibration\\*.jpg')
# # print(images)
# for fname in images:
#     img=cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # print(gray)

#     # Find the chess board corners
#     ret,corners = cv2.findChessboardCorners(gray,(7,6),None)
#     print(ret)

#     if ret == True:
#         objpoints.append(object_points)
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners2)
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img,(7,6),corners2,ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()
