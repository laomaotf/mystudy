from pathlib import Path
import fire
from sgfmill import sgf
import cv2
import numpy as np
import math
from tqdm import tqdm
from sklearn.cluster import KMeans

K = 500
MIN_MOVES = 100
MAX_CONTOUR = 19*19
USE_HU_MOMENT = True
MIN_CONTOUR = 3 if USE_HU_MOMENT else 15
def load_sgf(path):
    with open(path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    game_size = game.get_size()
    if game_size != 19:
        raise Exception("game_size != 19")
    moves = []
    for node in game.get_main_sequence():
        move = node.get_move()
        if (move[0] is None or move[1] is None):
            continue
        p,(y,x) = node.get_move()
        moves.append((p,(x,y)))
    if len(moves) < MIN_MOVES:
        raise Exception("len(moves) < 100")
    return game_size, moves

class REPLAYER(object):
    def __init__(self,game_size,moves,unit_size = 7) -> None:
        self.unit_size = unit_size
        self.stone_size = self.unit_size * 2 + 1
        self.block_size = int(self.stone_size * 2.5)
        self.block_count = game_size
        self.board_size = (self.block_count -1)  * self.block_size
        self.boundary_size = self.block_size
        self.moves = moves
        self.board  = self._reset_board()
        self.current = -1
    def _reset_board(self):
        self.board = np.zeros((self.board_size + self.boundary_size*2, self.board_size + self.boundary_size*2,3),np.uint8)
        self.board[:,:,1:] = 255
        for k in range(self.block_count):
            y = k * self.block_size
            cv2.line(self.board,(self.boundary_size,y+self.boundary_size),(self.board_size+self.boundary_size,y+self.boundary_size),
                     (0,0,0),self.unit_size//2)
        for k in range(self.block_count):
            x = k * self.block_size
            cv2.line(self.board,(x+self.boundary_size,self.boundary_size),(x+self.boundary_size,self.boundary_size+self.board_size),
                    (0,0,0),self.unit_size//2)
        return self.board
    def stepall(self):
        N = len(self.moves)
        self.step(N)
        return 
    def step(self,N=1):
        for _ in range(N):
            self.current = self.current + 1 if self.current + 1 < len(self.moves) else  len(self.moves) - 1
            color,(x,y) = self.moves[self.current]
            x *= self.block_size
            y *= self.block_size
            if color == "b":
                cv2.circle(self.board,(x+self.boundary_size,y+self.boundary_size),self.stone_size,(0,0,0),-1)
            else:
                cv2.circle(self.board,(x+self.boundary_size,y+self.boundary_size),self.stone_size,(255,255,255),-1)
            if self.current == len(self.moves) - 1:
                break
        return
    def save(self,path):
        cv2.imwrite(path, self.board)
        return
    def show(self,wait=10):
        cv2.imshow("GO",self.board)
        cv2.waitKey(wait)
    def __len__(self):
        return len(self.moves)
        
        
class FEATURE(object):
    def __init__(self,game_size, moves) -> None:
        self.block_count = game_size
        self.moves = moves
        return 
    def get_object_features(self,img):
        if 0:
            moves = []
            Y,X = np.where(img==255)
            for x,y in zip(X,Y):
                moves.append(("b",(x,y))) 
            rply = REPLAYER(img.shape[0],moves)
            rply.stepall()
            rply.save("background_all.png")
        
        num,labels = cv2.connectedComponents(img,connectivity=4)
        contours = []
        for label in range(1,num):
            mask = (labels == label).astype(np.uint8) * 255
            cnts,_ = cv2.findContours(mask,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
            
            for cnt in cnts:
                contours.append(cnt.reshape(-1,2).tolist())
                
            if 0:         
                moves = []
                Y,X = np.where(mask==255)
                for x,y in zip(X,Y):
                    moves.append(("b",(x,y)))
                #for pts in cnts:
                #    pts = pts.reshape(-1,2)
                #    for (x,y) in pts:
                #        moves.append(("w",(x,y)))
                rply = REPLAYER(mask.shape[0],moves)
                rply.stepall()
                rply.save("cca.png")
        return contours
    def get_fsd(self, contour, N):
        N = N + 1
        assert len(contour) >= N
        ft = [0 for _ in range(N)]
        s = len(contour)
        for u in range(N):
            rv, iv = 0,0
            cx, cy = np.mean([xy[0] for xy in contour]), np.mean([xy[1] for xy in contour])
            cx,cy = 0,0
            for j,(x,y) in enumerate(contour):
                rad = 2*math.pi*u*j/s
                cosr, sinr = math.cos(rad), math.sin(rad)
                rv += (x-cx) * cosr + (y-cy) * sinr
                iv += (y-cy) * cosr - (x-cx) * sinr
            ft[u] = math.sqrt(rv*rv + iv * iv)
        ft = np.array(ft)
        ft = ft/(ft[0] + np.finfo("float").eps)
        return ft[1:].tolist()
    def get_hu_moment(self, contour, N):
        img = np.zeros((self.block_count,self.block_count),np.uint8)
        for (x,y) in contour:
            img[y,x] = 255
        moments = cv2.moments(img)
        humoments = cv2.HuMoments(moments)
        return humoments.flatten().tolist()
    def get(self):
        boundary = 0
        img = np.zeros((self.block_count+2*boundary,self.block_count+2*boundary),np.uint8)
        for _,(x,y) in self.moves:
            img[y+boundary,x+boundary] = 255
        contours = self.get_object_features(255-img)
        fts,cnts = [],[]
        for contour in contours:
            if len(contour) < MIN_CONTOUR or len(contour) > MAX_CONTOUR:
                continue
            if USE_HU_MOMENT:
                ft = self.get_hu_moment(contour,-1)
            else:
                ft = self.get_fsd(contour,MIN_CONTOUR-1)
            fts.append(ft)
            cnts.append(contour)
        return fts,cnts
    def __len__(self):
        return len(self.moves) 
            
def main(indir,outdir):
    Path(outdir).mkdir(exist_ok=True)
    paths = Path(indir).glob("**/*.sgf")
    feats, contours = [], []
    for _,path in tqdm(enumerate(paths)):
        try:
            game_size, moves = load_sgf(path)
        except Exception as e:
            print(e)
            continue
        if 0:
            rply = REPLAYER(game_size, moves)
            rply.step(len(rply))
            rply.save("map.png")
        feature = FEATURE(game_size,moves)
        fts,cnts = feature.get()
        feats.extend(fts)
        contours.extend(cnts)
    kms = KMeans(n_clusters=K,random_state=42)
    preds = kms.fit_predict(feats) 
    print("size of sample: ",len(preds))
    results = [[] for _ in range(K)] 
    for pred, feat, contour in zip(preds, feats, contours):
        results[pred].append((feat, contour))
    sizes = [len(res) for res in results]
    indices = np.argsort(sizes)[::-1]
    for idx,idx_size in tqdm(enumerate(indices)):
        pred_out = str(Path(outdir).joinpath(f"{idx:05d}"))
        result = results[idx_size]
        for idx_img,(feat, contour) in enumerate(result):
            moves = []
            for (x,y) in contour:
                moves.append(('b',(x,y)))
            rply = REPLAYER(game_size, moves)
            rply.step(len(rply))
            Path(pred_out).mkdir(exist_ok=True)
            path = str(Path(pred_out).joinpath(f"{idx_img}.png"))
            rply.save(path)
    return
        
if __name__=="__main__":
    fire.Fire(main)
        