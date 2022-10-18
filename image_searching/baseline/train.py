import torch
import os,sys
from dataiter import DATAITER_CLASS as DATA
from network import NETWORK, calc_loss
import yaml
import logging

def train(device,
          lr0,
          traindata_dir, step_num_train, 
          outdir,model_file,**kwargs):
    network = NETWORK()
    if model_file != "":
        network.load_state_dict(torch.load(model_file))
        print(f"loading model {model_file}")
    traindata_iter = DATA(traindata_dir,**kwargs) 
    os.makedirs(outdir,exist_ok=True)
    optimizer = torch.optim.SGD(network.parameters(), lr = lr0, weight_decay=5e-5,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_num_train, eta_min=1e-9, last_epoch=-1)
    network.to(device)
    network.train()
    train_log = []
    for step_index in range(step_num_train):
        batch_data = traindata_iter.next()
        batch_data = batch_data.to(device,dtype=torch.float32)
        loss = calc_loss(network(batch_data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_log.append(
            f"{step_index:>08d},{loss:.5f},{scheduler.get_last_lr()[0]}"
        )
        if step_index % 1000 == 0:
            torch.save(network.state_dict(),os.path.join(outdir,f"step_{step_index}.pth"))  
            with open(os.path.join(outdir,"train.log"),'w') as f:
                f.write('\n'.join(train_log))
    print("end of training")       
    return


            
if __name__ == "__main__":
    train("cuda",0.001,r"/val",
           80000, "run",batch_size=16,width=224,height=224,worker_num=3,
           model_file='step_31000.pth') 
        
        
         
    