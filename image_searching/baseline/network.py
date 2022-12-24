
import torch 
import numpy as np
import random
from torchvision.models import resnet18
import torch.nn.functional as F
__all__ = ['NETWORK']


class NETWORK(torch.nn.Module):
    def __init__(self,feat_dim=128,**kwargs):
        super().__init__()
        self.backbone = resnet18(pretrained=False,**kwargs) 
        in_plane = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_plane,feat_dim)
        return
    def forward(self,x):
        feats = self.backbone(x)
        norm = torch.sum(feats**2,dim=1,keepdim=True) + 1e-6
        feats = feats / torch.sqrt(norm)
        return feats
    


def triplet_loss(feats, margin=1.0,ord=2):
    K = feats.shape[0]//2
    feats_pos, feats_neg  = feats[0:K], feats[K:]
    dist_in_class, dist_out_class = [], []
    for n in range(K):
        din = feats_pos[n] - feats_pos
        dist_in_class.append(din)
        dout = feats_pos[n] - feats_neg
        dist_out_class.append(dout)
    dist_in_class,dist_out_class = torch.cat(dist_in_class), torch.cat(dist_out_class)
    dist_in_class = torch.linalg.norm(dist_in_class,ord=ord,dim=1)
    dist_out_class = torch.linalg.norm(dist_out_class,ord=ord,dim=1)
    
    dist_pos, dist_neg = [],[] 
    for n in range(K):
        n0,n1 = n * K, (n+1) * K
        dp = torch.unsqueeze(dist_in_class[n0:n1].max(),dim=0)
        dn = torch.unsqueeze(dist_out_class[n0:n1].min(),dim=0)
        dist_pos.append(dp)
        dist_neg.append(dn)
    dist_pos,dist_neg = torch.cat(dist_pos), torch.cat(dist_neg)
    loss = dist_pos - dist_neg + margin
    return torch.relu(loss).mean()



def triplet_loss_ranking(feats, margin=0.3,ord=2):
    K = feats.shape[0]//2
    feats_pos, feats_neg  = feats[0:K], feats[K:]
    dist_in_class, dist_out_class = [], []
    for n in range(K):
        dist_in_class.append(feats_pos[n] - feats_pos)
        dist_out_class.append(feats_pos[n] - feats_neg)
    dist_in_class,dist_out_class = torch.cat(dist_in_class), torch.cat(dist_out_class)
    dist_in_class = torch.linalg.norm(dist_in_class,ord=ord,dim=1)
    dist_out_class = torch.linalg.norm(dist_out_class,ord=ord,dim=1)
    
    dist_pos, dist_neg = [],[] 
    for n in range(K):
        n0,n1 = n * K, (n+1) * K
        dist_pos.append(torch.unsqueeze(dist_in_class[n0:n1].max(),dim=0))
        dist_neg.append(torch.unsqueeze(dist_out_class[n0:n1].min(),dim=0))
    dist_pos,dist_neg = torch.cat(dist_pos), torch.cat(dist_neg)
    y = torch.ones_like(dist_neg)
    loss = torch.nn.functional.margin_ranking_loss(dist_neg,dist_pos,y,margin=margin)
    return loss


def triplet_loss_semihard(feats, margin=0.3,ord=2):
    K = feats.shape[0]//2
    feats_pos, feats_neg  = feats[0:K], feats[K:]
    dist_in_class, dist_out_class = [], []
    for k in range(K):
        dist_in_class.append(feats_pos[k] - feats_pos)
        dist_out_class.append(feats_pos[k] - feats_neg)
    dist_in_class,dist_out_class = torch.cat(dist_in_class), torch.cat(dist_out_class)
    dist_in_class = torch.linalg.norm(dist_in_class,ord=ord,dim=1)
    dist_out_class = torch.linalg.norm(dist_out_class,ord=ord,dim=1)
    dist_in_class = torch.reshape(dist_in_class,(K,K))
    dist_out_class = torch.reshape(dist_out_class,(K,K))
   
    
    dist_pos, dist_neg = [],[] 
    for k in range(K):
        ap = dist_in_class[:,k].repeat((K,1))
        mask = torch.logical_and(dist_out_class > ap, dist_out_class < ap + margin )
        an = torch.masked_select(dist_out_class, mask)
        ap = torch.masked_select(ap,mask)
        dist_pos.append(torch.reshape(ap,(-1,)))
        dist_neg.append(torch.reshape(an,(-1,)))
           
    if dist_pos == []:
        print("use hard samples")
        for n in range(K):
            n0,n1 = n * K, (n+1) * K
            dist_pos.append(torch.unsqueeze(dist_in_class[n0:n1].max(),dim=0))
            dist_neg.append(torch.unsqueeze(dist_out_class[n0:n1].min(),dim=0)) 
    dist_pos,dist_neg = torch.cat(dist_pos), torch.cat(dist_neg)
    loss = dist_pos - dist_neg + margin
    return torch.relu(loss).mean()

def triplet_loss_ranking_semihard(feats, margin=0.3,ord=2):
    K = feats.shape[0]//2
    feats_pos, feats_neg  = feats[0:K], feats[K:]
    dist_in_class, dist_out_class = [], []
    for k in range(K):
        dist_in_class.append(feats_pos[k] - feats_pos)
        dist_out_class.append(feats_pos[k] - feats_neg)
    dist_in_class,dist_out_class = torch.cat(dist_in_class), torch.cat(dist_out_class)
    dist_in_class = torch.linalg.norm(dist_in_class,ord=ord,dim=1)
    dist_out_class = torch.linalg.norm(dist_out_class,ord=ord,dim=1)
    dist_in_class = torch.reshape(dist_in_class,(K,K))
    dist_out_class = torch.reshape(dist_out_class,(K,K))
   
    #dist_in_class = F.pairwise_distance(feats_pos, feats_pos, p=2)
    #dist_out_class = F.pairwise_distance(feats_pos, feats_neg, p=2)
    
    dist_pos, dist_neg = [],[] 
    for k in range(K):
        ap = dist_in_class[:,k].repeat((K,1))
        # select semi-hard sample for training
        mask = torch.logical_and(dist_out_class > ap, dist_out_class < ap + margin )
        an = torch.masked_select(dist_out_class, mask)
        ap = torch.masked_select(ap,mask)
        dist_pos.append(torch.reshape(ap,(-1,)))
        dist_neg.append(torch.reshape(an,(-1,)))
           
    if dist_pos == []:
        print("use hard samples")
        for n in range(K):
            n0,n1 = n * K, (n+1) * K
            dist_pos.append(torch.unsqueeze(dist_in_class[n0:n1].max(),dim=0))
            dist_neg.append(torch.unsqueeze(dist_out_class[n0:n1].min(),dim=0)) 
    dist_pos,dist_neg = torch.cat(dist_pos), torch.cat(dist_neg)
    y = torch.ones_like(dist_neg)
    loss = torch.nn.functional.margin_ranking_loss(dist_neg,dist_pos,y,margin=margin)
    return loss

def calc_loss(feats,loss_type="triplet_loss_ranking_semihard",**kwargs):
    if loss_type.lower() == "triplet_loss":
        return triplet_loss(feats,**kwargs)
    elif loss_type.lower() == "triplet_loss_semihard":
        return triplet_loss_semihard(feats,**kwargs)
    elif loss_type.lower() == "triplet_loss_ranking_semihard":
        return triplet_loss_ranking_semihard(feats,**kwargs)
    raise Exception(f"not supported loss {loss_type}")


if __name__ == "__main__":
    net = NETWORK(128,progress=True)
    X = np.random.uniform(low = -1, high = 1, size = (24,3,224,224)).astype(np.float32)
    X = torch.from_numpy(X)
    Y = net(X)
    loss = calc_loss(Y)
    print(f"X = {X.shape}")
    print(f"Y = {Y.shape}")
    print(f"loss = {loss.shape}")
    

    