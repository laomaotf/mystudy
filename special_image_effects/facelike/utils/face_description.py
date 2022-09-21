import dlib
import os,sys
import numpy as np
import cv2

class FaceDescription(object):
    def __init__(self,model_path=""):
        if model_path == "":
            model_path = os.path.join(os.path.dirname(__file__),"..","data","dlib_face_recognition_resnet_model_v1.dat")
        self.m_recognizer = dlib.face_recognition_model_v1(model_path)
        return 
    def forward_opencv_image(self,image,xyxy, points):
        desc = None
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isinstance(xyxy,(list,tuple)):
            rect = dlib.rectangle(*xyxy) 
            parts = []
            for (x,y) in points:
                part = dlib.point(x,y)
                parts.append(part)
            det = dlib.full_object_detection(rect,parts)
        else:
            return desc
        desc_raw = self.m_recognizer.compute_face_descriptor(image,det)
        if desc_raw is not None:
            desc = []
            for d in desc_raw:
                desc.append(d)
        return desc    
    
if __name__=="__main__":
    from face_detection import FaceDetection
    from face_landmark import FaceLandmark
    FD = FaceDetection()
    FL = FaceLandmark()
    FR = FaceDescription()
    img = cv2.imread('README/1.jpg',1)
    H,W,C = img.shape
    faces = FD.forward_opencv_image(img)
    shape = FL.forward_opencv_image(img,faces[0])
    desc = FR.forward_opencv_image(img,faces[0],shape)
    print(desc)