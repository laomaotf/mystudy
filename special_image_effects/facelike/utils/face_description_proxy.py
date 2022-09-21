import os,sys 
import cv2 
import numpy as np 
from utils.face_detection import FaceDetection 
from utils.face_landmark import FaceLandmark
from utils.face_description import FaceDescription
import pickle

class FaceDescriptionProxy(object):
    def __init__(self):
        self.m_detector = FaceDetection()
        self.m_predictor_landmark = FaceLandmark()
        self.m_predictor_desc = FaceDescription()
        return 
    def __call__(self, img_path):
        desc_path = img_path + ".ft"
        if os.path.exists(desc_path):
            with open(desc_path,'rb') as f:
                desc = pickle.load(f)
            return desc
        img = cv2.imread(img_path,1)
        faces = self.m_detector.forward_opencv_image(img)
        shape = self.m_predictor_landmark.forward_opencv_image(img,faces[0])
        desc = self.m_predictor_desc.forward_opencv_image(img,faces[0],shape)
        with open(desc_path,'wb') as f:
            pickle.dump(desc,f)
        return desc
    

if __name__ == "__main__":
    FDProxy = FaceDescriptionProxy()
    img_path = "README/1.jpg"    
    desc = FDProxy(img_path)
    print(desc)
    