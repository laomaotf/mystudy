import dlib
import os,sys
import numpy as np
import cv2

class FaceLandmark(object):
    m_predictor_ = None
    def __init__(self, predictor_path = ""):
        if predictor_path == "":
            predictor_path = os.path.join(os.path.dirname(__file__),"..","data","shape_predictor_68_face_landmarks.dat")    
        self.load(predictor_path)
    def load(self,predictor_path):
        self.m_predictor = dlib.shape_predictor(predictor_path)
        return self.m_predictor
    def forward_opencv_image(self,image, xyxy):
        points = []
        if self.m_predictor is None:
            return points
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box = dlib.rectangle(*xyxy)
        shape = self.m_predictor(image,box)
        for part in shape.parts():
            x,y = part.x,part.y 
            points.append((x,y))
        return points
    
if __name__=="__main__":
    from face_detection import FaceDetection
    FD = FaceDetection()
    FL = FaceLandmark("data/shape_predictor_68_face_landmarks.dat")
    img = cv2.imread('README/1.jpg',1)
    H,W,C = img.shape
    faces = FD.forward_opencv_image(img)
    points = FL.forward_opencv_image(img,faces[0])
    for xy in points:
        cv2.circle(img,xy, 3, (0,255,0))
    cv2.imshow("FL",img)
    cv2.waitKey(4000)