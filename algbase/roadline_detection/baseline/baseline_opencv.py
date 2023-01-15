import cv2
import os,sys
import numpy as np
import copy
import math


def doSegmentation(image_data,th0=50, th1=100):
    seg_data = cv2.Canny(image_data, th0,th1) 
    return seg_data

def removeNoiseRegion(seg_data):
    mask = np.zeros_like(seg_data)
    roadline_contour = np.array([[305,658],[613,432],[663,432],[1008,658]],dtype=np.int)
    mask = cv2.fillPoly(mask,[roadline_contour],color=255)
    seg_data = cv2.bitwise_and(mask,seg_data)
    return seg_data

def calcSlope(line):
    x0,y0,x1,y1 = line[0]
    return (y1-y0) / (x1-x0+0.00001)


def findTwoLines(seg_data):
    lines = cv2.HoughLinesP(seg_data,1, np.pi/180,20,minLineLength=50, maxLineGap=30)
    llines = list(filter(lambda l: calcSlope(l) < 0, lines))
    rlines = list(filter(lambda l: calcSlope(l) > 0, lines))
    return llines, rlines

def removeNoiseLine(lines,stdvarT=0.1):
    slopes = list(map(lambda l: calcSlope(l), lines))
    slopes = np.array(slopes)
    kept = [k for k in range(len(lines))]
    while(kept != []):
        mean = slopes[kept].mean()
        stdvars  = np.abs(slopes[kept] - mean)
        k = np.argmax(stdvars)
        if stdvars[k] < stdvarT:
            break
        else:
            kept = list(set(kept) - set([kept[k]]) )
    return [lines[k] for k in kept]
             
def fitLine(lines):
    xall,yall = [],[]
    for line in lines:
        x0,y0,x1,y1 = line[0]
        xall.extend([x0,x1])
        yall.extend([y0,y1])
    coef = np.polyfit(xall,yall,deg=1)
    x0,x1 = np.min(xall),np.max(xall)
    y0,y1 = np.polyval(coef,x0), np.polyval(coef,x1)
    return np.array([x0,y0,x1,y1],dtype=np.int).reshape(-1)
    
def doScanDir(frames_folder):
    paths = filter(lambda x: os.path.splitext(x)[-1].lower() == '.jpg',  os.listdir(frames_folder))
    paths = map(lambda x: os.path.join(frames_folder,x), list(paths))
    return list(paths)

def drawLines(image_data, lines,color,thickness=1):
    if len(image_data.shape) != 3:
        color_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
    else:
        color_data = copy.deepcopy(image_data)
    for line in lines:
        x0,y0,x1,y1 = line[0].tolist()
        cv2.line(color_data, (x0,y0),(x1,y1),color=color,thickness=thickness)
    return color_data

def main():
    print("opencv:",cv2.__version__)
    paths = doScanDir(os.path.join(os.path.dirname(__file__),"images"))
    for path in paths:
        frame = cv2.imread(path,0)
        seg_data = doSegmentation(frame)
        cv2.imshow("roadline",seg_data),cv2.waitKey(1000)
        seg_data = removeNoiseRegion(seg_data)
        cv2.imshow("roadline",seg_data), cv2.waitKey(1000)
        llines,rlines = findTwoLines(seg_data)
        color_data = drawLines(frame,llines, (0,0,255))
        color_data = drawLines(color_data,rlines, (255,0,0)) 
        cv2.imshow("roadline",color_data),cv2.waitKey(1000)
        
        llines = removeNoiseLine(llines,0.1)
        rlines = removeNoiseLine(rlines,0.1)
        color_data = drawLines(frame,llines, (0,0,255))
        color_data = drawLines(color_data,rlines, (255,0,0)) 
        cv2.imshow("roadline",color_data),cv2.waitKey(1000)
        
        lline = fitLine(llines)
        rline = fitLine(rlines)
        color_data = drawLines(frame,[[lline]], (0,0,255),thickness=3)
        color_data = drawLines(color_data,[[rline]], (255,0,0),thickness=3) 
        cv2.imshow("roadline",color_data),cv2.waitKey(1000)
        
        cv2.waitKey(-1)
    
if __name__=="__main__":
    main() 
        
