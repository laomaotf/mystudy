import numpy as np 
import cv2
import os,sys
import copy

#mask范围内，相邻位置设置成1，对角线设置成-4
#这里其实只有A矩阵中包含未知量的一部分，另一部分已知量(边缘)被融合到B中
def createA(mask,xy_unk,xy2ind):
    H,W = mask.shape[0:2]
    A = np.eye(len(xy_unk),dtype=np.float32) * (-4)
    for (ay,ax) in xy_unk:
        a = xy2ind[(ay,ax)]
        for dy in range(-1,2):
            for dx in range(-1,2):
                if dx == 0 and dy == 0:
                    continue
                if dx != 0 and dy != 0:
                    continue
                x,y = ax + dx, ay+dy
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue
                if mask[y,x] == 0:
                    continue
                b = xy2ind[(y,x)]
                A[a,b] = 1
    img = np.clip((A + 6) * 30,0,255).astype(np.uint8)
    cv2.imwrite("A.png",img)
    return A

def createB(mask,foreground,background,offset,xy_unk,xy2ind):
    grad = cv2.Laplacian(foreground,cv2.CV_32FC1)
    cv2.imwrite("grad.png",np.clip(np.abs(grad),0,255).astype(np.uint8))
    H,W = foreground.shape[0:2]
    B = np.zeros((len(xy_unk),1),dtype=np.float32)
    for (ay,ax) in xy_unk: 
        v = 0
        a = xy2ind[(ay,ax)]
        #以下实际是A矩阵中的已知量部分(被移动到等号右侧，归入B的部分)
        for dy in range(-1,2):
            for dx in range(-1,2):
                if dx == 0 and dy == 0:
                    continue
                if dx != 0 and dy != 0:
                    continue
                x,y = ax + dx, ay + dy
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue
                if mask[y,x] != 0:
                    continue #(ax,ay)位于边缘位置
                v += background[y+offset[1],x+offset[0]]
        B[a,0] = grad[ay,ax] - v
    return B

def possion_editing(mask,foreground,background,offset):
    assert(3 == foreground.shape[-1])," only 3-channels image supported"
    H,W = mask.shape[0:2]
    xy_unk = []
    xy2ind = {}
    for y in range(H):
        for x in range(W):
            if mask[y,x] == 0:
                continue
            xy_unk.append((y,x))
            xy2ind[(y,x)] = len(xy_unk) - 1
    matA = createA(mask, xy_unk,xy2ind)
    merged = copy.deepcopy(background)
    for c in range(3): 
        matB = createB(mask,foreground[:,:,c],background[:,:,c],offset,xy_unk,xy2ind)
        pixels_merged = np.linalg.solve(matA, matB)
        pixels_merged = np.clip(pixels_merged,0,255).astype(np.uint8)
        merged_roi = merged[offset[1]:offset[1]+H, offset[0]:offset[0]+W,c]
        for k,(y,x) in enumerate(xy_unk):
            merged_roi[y,x] = pixels_merged[k]
        cv2.imwrite(f"merged_small_{c}.png",merged_roi)
    cv2.imwrite("foreground.png",foreground)
    cv2.imwrite("background.png",background)
    return merged

if __name__ == "__main__":
    foreground = cv2.imread("f.jpg",1)
    foreground = cv2.resize(foreground,(50,50))
    background = cv2.imread("g.jpg",1)
    H,W = foreground.shape[0:2]
    mask = np.zeros((H,W),dtype=np.uint8)
    mask[2:-2,2:-2] = 255
    merged = possion_editing(mask,foreground,background,(50,50))
    cv2.imwrite("merged.png",merged)
        
            
            
    
                    
                    
                    
    
    