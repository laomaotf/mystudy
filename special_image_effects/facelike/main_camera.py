import cv2
import numpy as np  
from utils.face_detection import FaceDetection
from utils.face_landmark import FaceLandmark
import utils.warp_image as FaceWarping
import warnings
from queue import Queue 
from threading import Thread
import sys
def preprocess_test(frame):
    FD = FaceDetection()
    FL = FaceLandmark()
    faces = FD.forward_opencv_image(frame)
    if len(faces) < 1:
        warnings.warn("no face found")
        return frame
    landmarks = FL.forward_opencv_image(frame,faces[0])
    for (x,y) in landmarks:
        cv2.circle(frame,(x,y),3,(0,255,0),3)
    return frame

def warp_faces(tasksQ):
    print("thread start")
    while 1: 
        args = tasksQ.get(block=True)
        frame,foreground = args
        frame = cv2.resize(frame,(512//1,512//1))
        foreground = cv2.resize(foreground,(512//1,512//1))
        warpped = FaceWarping.main(foreground,frame,0)
        if warpped is not None:
            cv2.imshow("fake",warpped)
    print('thread exit')
    return

def main(foreground_path):
    foreground = cv2.imread(foreground_path,1)
    taskQ = Queue(maxsize=1024)
    for n in range(10):
        Thread(target=warp_faces,args=(taskQ,)).start()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("fake")
    cv2.namedWindow("src")
    frame_num = 0
    while cv2.waitKey(1) == -1:
        frame_num += 1
        _, frame = cap.read()
        frame = cv2.flip(frame,1)
        taskQ.put([frame,foreground],block=False)
        #frame = preprocess_test(frame)
        cv2.imshow("src",frame)
        #cv2.waitKey(50)
    cap.release()
    taskQ.queue.clear()
    cv2.destroyAllWindows()
    
    
if __name__=="__main__":
    main(sys.argv[1])