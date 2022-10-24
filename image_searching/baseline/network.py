import torch 
from torchvision.models import resnet18
__all__ = ['NETWORK']


class NETWORK(torch.nn.Module):
    def __init__(self,feat_dim=128,**kwargs):
        super().__init__()
        self.backbone = resnet18(pretrained=True,**kwargs) 
        in_plane = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_plane,feat_dim)
        return
    def forward(self,x):
        feats = self.backbone(x)
        return feats
    
def triplet_loss(feats, margin=1.0):
    feats_anchor, feats_pos, feats_neg  = feats[0::3], feats[1::3], feats[2::3]
    dist_in_class = torch.square(feats_anchor - feats_pos).sum(axis=-1,keepdim=True)
    dist_out_class = torch.square(feats_anchor - feats_neg).sum(axis=-1,keepdim=True)
    loss = dist_in_class - dist_out_class + margin
    return torch.relu(loss).mean()


    
        
def calc_loss(feats,loss_type="triplet_loss",**kwargs):
    if loss_type.lower() == "triplet_loss":
        return triplet_loss(feats, **kwargs)
    raise Exception(f"not supported loss {loss_type}")


if __name__ == "__main__":
    net = NETWORK(128,progress=True)
    X = torch.zeros(size=(24,3,224,224))
    Y = net(X)
    loss = calc_loss(Y)
    print(f"X = {X.shape}")
    print(f"Y = {Y.shape}")
    print(f"loss = {loss.shape}")
    

    