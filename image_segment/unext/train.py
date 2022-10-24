import argparse
import os
from collections import OrderedDict
from glob import glob

import albumentations.augmentations.transforms
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import absolute_path, AverageMeter


def update_output_dir(config):
    dataset_name = config['dataset_dir'].replace('\\','/').strip('/').split('/')[-1]
    training_name = '%s_%s' % (dataset_name, config['arch'])
    current_output_dir = os.path.join(config['output_dir'],'%s' % training_name)
    indices = [-1]
    for name_index in os.listdir(config['output_dir']):
        name = '_'.join(name_index.split('_')[0:-1])
        index = name_index.split('_')[-1]
        if name == training_name:
            indices.append(int(index) )
    current_output_dir = os.path.join(config['output_dir'],'%s_%d' % (training_name,max(indices) +1 ))
    config['output_dir'] = current_output_dir
    return current_output_dir
                                   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='yaml configure file path')
    args = parser.parse_args()
    cfg_path = absolute_path(args.cfg)
    with open(cfg_path,'r') as f:
        config = yaml.safe_load(f)
    config['dataset_dir'] = absolute_path(config['dataset_dir'])
    config['output_dir'] = absolute_path(config['output_dir'])
    update_output_dir(config)
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    return config

# args = parser.parse_args()
def train(epoch,config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  "dice":AverageMeter()}

    model.train()

    for iter,(input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters["dice"].update(dice,input.size(0))
        if iter % 100 == 0:
            print(f"epoch {epoch} iter {iter} loss {avg_meters['loss'].avg:.3f} iou {avg_meters['iou'].avg:.3f} dice {avg_meters['dice'].avg:.3f}")

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)
                        ])


def validate(epoch,config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for iter, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters["dice"].update(dice,input.size(0))
            if iter % 100 == 0:
                print(f"epoch {epoch} iter {iter} loss {avg_meters['loss'].avg:.3f} iou {avg_meters['iou'].avg:.3f} dice {avg_meters['dice'].avg:.3f}")

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

   
        

def main():
    config = parse_args()
    os.makedirs(config['output_dir'], exist_ok=True)
    images_dir = os.path.join(config['dataset_dir'],"images")
    masks_dir = os.path.join(config['dataset_dir'],"masks")
    

    with open(os.path.join(config['output_dir'], 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join(images_dir, '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        RandomRotate90(),
        albumentations.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(max_pixel_value=255.0),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(max_pixel_value=255.0),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=images_dir,
        mask_dir=masks_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=images_dir,
        mask_dir=masks_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        # train for one epoch
        train_log = train(epoch,config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(epoch,config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'],
                 val_log['loss'], val_log['iou'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(scheduler.get_last_lr()[0])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(os.path.join(config["output_dir"],'log.csv'), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(config["output_dir"],'best.pth'))
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
            
        torch.save(model.state_dict(), os.path.join(config["output_dir"],'final.pth'))
        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
