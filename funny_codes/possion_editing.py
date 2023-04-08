import numpy as np 
import cv2
import os,sys
import copy
import argparse

SHOW_INTER_IMAGE = False



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
    if SHOW_INTER_IMAGE:
        img = np.clip((A + 6) * 30,0,255).astype(np.uint8)
        cv2.imwrite("A.png",img)
    return A

def get_mixing_gradient(imga, imgb):
    kernels = [
        (0,1,0,  0,-1,0,  0,0,0),
        (0,0,0,  1,-1,0,  0,0,0),
        (0,0,0,  0,-1,1,  0,0,0),
        (0,0,0,  0,-1,0,  0,1,0),
    ]
    H,W = imga.shape[0:2]
    graya = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    grayb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    grad = np.zeros((H,W),dtype=np.float32)
    for kernel in kernels:
        grada = cv2.filter2D(graya,cv2.CV_32FC1,np.reshape(kernel,(3,3)))
        gradb = cv2.filter2D(grayb,cv2.CV_32FC1,np.reshape(kernel,(3,3)))
        grad += np.where(np.abs(grada) > np.abs(gradb), grada, gradb)
    grad = np.expand_dims(grad,axis=2)
    return np.concatenate([grad,grad,grad],axis=2) 
    
def get_gradient(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = np.expand_dims(cv2.Laplacian(gray,cv2.CV_32FC1),axis=2)
    return np.concatenate([grad,grad,grad],axis=2)

def createB(mask,foreground,background,offset,xy_unk,xy2ind,use_mix_grad, grad_smooth):
    H,W, C = foreground.shape
    if 1:
        grad_fg = get_gradient(foreground)
    else:
        #grads_fg = [np.expand_dims(cv2.Laplacian(foreground[:,:,c],cv2.CV_32FC1),axis=-1) for c in range(C)]
        gray_fg = cv2.cvtColor(foreground,cv2.COLOR_BGR2GRAY)
        grads_fg = [np.expand_dims(cv2.Laplacian(gray_fg,cv2.CV_32FC1),axis=-1) for c in range(C)]
        grad_fg = np.concatenate(grads_fg,axis=2)
        
    if use_mix_grad:
        grad = get_mixing_gradient(foreground,background[offset[1]:offset[1]+H,offset[0]:offset[0]+W])
    else:
        grad = grad_fg
        
    grad_mag = np.sum(np.abs(grad),axis=-1,keepdims=True) / 255.0
    grad = np.where(grad_mag > grad_smooth, grad, 0)
        
    if SHOW_INTER_IMAGE:
        for c in range(C):
            cv2.imwrite(f"grad_channel{c}.png",np.clip(np.abs(grad[:,:,c]),0,255).astype(np.uint8))
    B = np.zeros((len(xy_unk),C),dtype=np.float32)
    for (ay,ax) in xy_unk: 
        vs = [0,0,0]
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
                vs[0] += background[y+offset[1],x+offset[0],0]
                vs[1] += background[y+offset[1],x+offset[0],1]
                vs[2] += background[y+offset[1],x+offset[0],2]
        B[a,0] = grad[ay,ax,0] - vs[0]
        B[a,1] = grad[ay,ax,1] - vs[1]
        B[a,2] = grad[ay,ax,2] - vs[2]
    return B

def possion_editing(mask,foreground,background,offset,use_mix_grad,smooth_grad):
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
    matB = createB(mask,foreground,background,offset,xy_unk,xy2ind,use_mix_grad,smooth_grad)
    pixels_merged = np.linalg.solve(matA, matB)
    pixels_merged = np.clip(pixels_merged,0,255).astype(np.uint8)
    
    #copy 3-channels
    merged_roi = merged[offset[1]:offset[1]+H, offset[0]:offset[0]+W]
    for k,(y,x) in enumerate(xy_unk):
        merged_roi[y,x] = pixels_merged[k]
        
    return merged

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fg",help="foregroud image")
    ap.add_argument("bg",help="background image")
    ap.add_argument('-mg',"--mix-grad",action="store_true")
    ap.add_argument('-sg',"--smooth-grad",default=0,type=np.float32)
    args = ap.parse_args()
    print(args.__dict__)
    assert(args.smooth_grad >= 0 and args.smooth_grad <= 1)
    foreground = cv2.imread(args.fg,1)
    foreground = cv2.resize(foreground,(100//2,100//2))
    background = cv2.imread(args.bg,1)
    H,W = foreground.shape[0:2]
    mask = np.zeros((H,W),dtype=np.uint8)
    mask[2:-2,2:-2] = 255
    
   
    # mergedCV = cv2.seamlessClone(foreground,background,mask,(50,50),cv2.NORMAL_CLONE) 
    # cv2.imwrite("mergedCV_NORMAL.png",mergedCV)
    # mergedCV = cv2.seamlessClone(foreground,background,mask,(50,50),cv2.MIXED_CLONE) 
    # cv2.imwrite("mergedCV_MIXED.png",mergedCV)
    
    merged = possion_editing(mask,foreground,background,(50,50),args.mix_grad,args.smooth_grad)
    cv2.imwrite("merged.png",merged)
        
            
            
    
                    
                    
                    
    
    