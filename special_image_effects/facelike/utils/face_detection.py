import dlib
import os,sys
import numpy as np
import cv2

class FaceDetection(object):
    def __init__(self):
        self.m_detector = dlib.get_frontal_face_detector()
        return 
    def forward_opencv_image(self,image):
        faces = []
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = self.m_detector(image,1)
        for det in dets:
            x0,y0,x1,y1 = det.left(),det.top(),det.right(),det.bottom()
            faces.append((x0,y0,x1,y1))
        return faces 
    
if __name__=="__main__":
    FD = FaceDetection()
    img = cv2.imread('README/1.jpg',1)
    H,W,C = img.shape
    faces = FD.forward_opencv_image(img)
    for (x0,y0,x1,y1) in faces:
        cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),3)
    cv2.imshow("FD",img)
    cv2.waitKey(4000)