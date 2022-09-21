import cv2
import numpy as np
from utils.face_alignment import FaceAlignment
from utils.face_detection import FaceDetection
from utils.face_landmark import FaceLandmark
import copy
import warnings

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

MASK_GROUPS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

def make_mixture(img, img_ref):
    R = int(max(img.shape) * 0.4)
    R = R // 2 * 2 + 1
    img_blur = cv2.GaussianBlur(img,(R,R),0,0).astype(np.float32)
    img_ref_blur = cv2.GaussianBlur(img_ref,(R,R),0,0).astype(np.float32)
    #让img的色调和img_ref保持一致
    return np.clip(img * img_ref_blur / img_blur,0,255).astype(np.uint8)
    
def make_mask(img, landmarks,R=41):
    H,W,_ = img.shape
    mask = np.zeros((H,W),dtype=np.uint8)
    for group in MASK_GROUPS:
        pts = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(mask,pts,color=255)
    mask = cv2.GaussianBlur(mask,(R,R),0,0)
    return mask 

def draw_landmark(img, landmarks):
    img_show = copy.deepcopy(img)
    for (x,y) in landmarks:
        cv2.circle(img_show,(x,y), 3,(0,255,0),1)
    return img_show

def main(img0, img1,flag = 0):#paster img0 on img1
    FD = FaceDetection()
    FL = FaceLandmark()
    FA = FaceAlignment()

    faces0 = FD.forward_opencv_image(img0)
    if faces0 == []:
        warnings.warn("no face found in img0")
        return None
    landmarks0 = FL.forward_opencv_image(img0,faces0[0])
    landmarks0 = np.array(landmarks0)
    
    faces1 = FD.forward_opencv_image(img1)
    if faces1 == []:
        warnings.warn("no face found in img1")
        return None
    landmarks1 = FL.forward_opencv_image(img1,faces1[0])
    landmarks1 = np.array(landmarks1)


    #把img0矫正到和img1一致
    img0_warp = FA(img0,landmarks0,landmarks1)
    if flag != 0:
        cv2.imshow("img0",draw_landmark(img0,landmarks0))
        cv2.imshow("img1",draw_landmark(img1,landmarks1))
        cv2.imshow("img0_warp",img0_warp)
    img0_warp_mixed = make_mixture(img0_warp, img1)
    if flag != 0:
        cv2.imshow("img0_warp_mixed",img0_warp_mixed)


    #两个mask取并集，避免landmark的轻微差异(mask做了边缘羽化效果)
    mask0 = make_mask(img0, landmarks0)
    mask1 = make_mask(img1, landmarks1)
    mask0_warp = FA(mask0, landmarks0, landmarks1)
    mask = np.max([mask0_warp, mask1],axis=0)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if flag != 0:
        cv2.imshow("mask",mask) 
   
    #融合
    mask = mask / 255.0 
    img_res = img0_warp_mixed * mask + img1 * (1.0-mask)
    img_res = np.clip(img_res,0,255).astype(np.uint8)
    if flag != 0: 
        cv2.imshow("img_res",img_res) 
        cv2.waitKey(-1)
        
    return img_res

if __name__ == "__main__":
    img0 = cv2.imread("README/5.jpg",1)
    img1 = cv2.imread("README/4.jpg",1)
    img0 = cv2.resize(img0,(512,512))
    img1 = cv2.resize(img1,(512,512))
    main(img0,img1)