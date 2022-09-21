import cv2
import os,sys
import numpy as np   
from skimage import transform
class FaceAlignment(object):
    def __init__(self,eps = 0.0001):
        self.m_eps = eps
        return 
    def calcM(self, points_src, points_dst):
        pts0,pts1 = np.array(points_src).astype(np.float32), np.array(points_dst).astype(np.float32)
        tform = transform.SimilarityTransform()
        tform.estimate(pts0, pts1) 
        M = tform.params[0:2,:]
        return M
    def __call__(self, img_src, points_src, points_dst):
        M = self.calcM(points_src, points_dst)
        img_warp = np.zeros(img_src.shape,dtype=img_src.dtype)
        H,W = img_warp.shape[0:2]
        cv2.warpAffine(img_src,M,(W,H),img_warp,borderMode=cv2.BORDER_TRANSPARENT)
        return img_warp 
                 

if __name__=="__main__":
    from face_detection import FaceDetection
    from face_landmark import FaceLandmark
    FD = FaceDetection()
    FL = FaceLandmark("data/shape_predictor_68_face_landmarks.dat")
    FA = FaceAlignment()
    img1 = cv2.imread('README/1.jpg',1)
    faces1 = FD.forward_opencv_image(img1)
    points1 = FL.forward_opencv_image(img1,faces1[0])
    img2 = cv2.imread("README/2.jpg",1)
    faces2 = FD.forward_opencv_image(img2)
    points2 = FL.forward_opencv_image(img2,faces2[0])
  
    M1to2 = FA(points1,points2) 
    #img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    img3 = np.zeros(img1.shape,dtype=img1.dtype)
    H,W = img3.shape[0:2]

    cv2.warpAffine(img2,M1to2,(W,H),img3,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.WARP_INVERSE_MAP)
    cv2.imshow("img1",img1)
    cv2.imshow("img2",img2)
    cv2.imshow("results",img3)
    cv2.waitKey(-1) 
    
     
    