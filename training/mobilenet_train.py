import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torch.quantization import quantize_dynamic
import matplotlib.pyplot as plt
import random
import timeit
import argparse
import numpy as np
import os, sys
import random
from glob import glob
import os.path as osp
sys.path.append('../gmflow')

from data import build_train_dataset
from gmflow.gmflow import GMFlow
from loss import ttt_loss_func_general
from evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                      create_sintel_submission, create_kitti_submission, inference_on_dir)
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from PIL import Image
from loss import ttt_loss_func

from torch.utils.tensorboard import SummaryWriter
from utils.logger import Logger
import pylab as pl
from IPython import display
from utils.data_utils import SimulatedDataset, RGBDDataset
from utils.data_utils import normalize_img

# Specify the GPU device
gpu_no = 0
assert torch.cuda.is_available()
device = torch.device(f'cuda:{gpu_no}')
print(f"GPU {gpu_no}: {torch.cuda.get_device_name(gpu_no)}")

def build_dataset(stage, split, envs=None):
    if stage == 'simulated':
        #aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        
        train_dataset = SimulatedDataset(aug_params=None, split=split, envs=envs,  seq_idx_start=50)
    elif stage == 'rgbd':
        train_dataset = RGBDDataset(aug_params=None, split=split, envs=envs,  seq_idx_start=50)
    else:
        raise ValueError(f'stage {stage} is not supported')
    return train_dataset

def weighted_mse_loss(predicted, target, weight):
    loss_ttt = (torch.sum((predicted[:,0,:,:] - target[:,0,:,:])**2))
    loss_conf = (torch.sum((predicted[:,1,:,:] - target[:,1,:,:])**2))
    return weight[0]*loss_ttt + weight[1]*loss_conf

## general
save = True
finetune_kitti = False
finetune_rgbd = True
mobilev3_S = False
mobilev3_L = True
stage = 'rgbd'
resume = False
lr = 1e-3
if mobilev3_S:
    checkpoint_dir = './runs/mobilenetV3Small'
elif mobilev3_L:
    checkpoint_dir = './runs/mobilenetV3Large'
else:
    assert "Error, Wrong network!"
summary_freq = 100
weight_decay = 1e-4
total_epochs = 70

assert np.sum(mobilev3_S+mobilev3_L) == 1, "Only one network can be selected"

#envs_train_simulation = ['arena', 'cafe_dynamic', 'cafe_dynamic_2', 'corridor_dynamic', 'corridor_dynamic_brick', 'corridor_right', 'corridor_rightCorner', 'cyberzoo', 'fetchit', 'indoor_dynamic', 'LCorridor', 'new_indoor', 'new_indoor_2', 'new_indoor_dynamic', 'office', 'office_dynamic', 'office_dynamic_brick', 'office_env_large', 'office_large', 'office_small', 'office_small_2', 'small_city', 'small_city_2', 'small_city_3', 'vineyard_grass', 'vineyard_grass_2', 'vineyard_mix']
envs_train_simulation = ['arena', 'cafe_dynamic', 'cafe_dynamic_2', 'corridor_dynamic', 'corridor_dynamic_brick', 'corridor_right', 'corridor_rightCorner', 'cyberzoo', 'fetchit', 'indoor_dynamic', 'LCorridor', 'new_indoor', 'new_indoor_2', 'new_indoor_dynamic', 'office', 'office_dynamic', 'office_dynamic_brick', 'office_env_large', 'office_large', 'office_small', 'office_small_2', 'small_city_2', 'small_city_3', 'vineyard_grass', 'vineyard_grass_2', 'vineyard_mix']
envs_train_kitti = ['test1', 'test3', 'test4', 'test6', 'test8', 'test10', 'test12', 'test13', 'test15', 'test16', 'test19', 'test20', 'test23']
envs_train_rgbd = ['large_noloop', 'office_household', 'pioneer_360', 'pioneer_slam', 'pioneer_slam2', 'structure_notexture_far', 'structure_texture_far', 'room']
#envs_test_simulation = ['cafe', 'office', 'small_city', 'bookstore', 'bookstore_2', 'zigzag']
envs_test_simulation = ['cafe', 'office', 'bookstore', 'bookstore_2', 'zigzag']
envs_test_kitti = ['test2', 'test5', 'test7', 'test9', 'test11', 'test14', 'test17', 'test18', 'test21', 'test22']
envs_test_rgbd = ['pioneer_slam3', 'desk', 'desk2']

if finetune_kitti:
    envs_train = envs_train_kitti
    envs_test = envs_test_kitti
elif finetune_rgbd:
    envs_train = envs_train_rgbd
    envs_test = envs_test_rgbd
else:
    envs_train = envs_train_simulation
    envs_test = envs_test_simulation

train_dataset = build_dataset(stage, 'training', envs=envs_train)
test_dataset = build_dataset(stage, 'testing', envs=envs_test)

BATCH_SIZE = 8
NUM_WORKERS = 8

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS,
                                            pin_memory=True, drop_last=True,
                                            sampler=None)
print(f'Training images: {len(train_loader)*BATCH_SIZE}')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS,
                                            pin_memory=True, drop_last=True,
                                            sampler=None)
print(f'Testing images: {len(test_loader)*BATCH_SIZE}')

debug_dimensions = False

if mobilev3_S:
    mobilenet = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
elif mobilev3_L:
    mobilenet = mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V2')
else:
    assert "Error, Wrong network!"
    
mobilenet.features[0][0] = torch.nn.Conv2d(6, 16, kernel_size=7, stride=2, padding=3, bias=False)

if mobilev3_S:
    mobilenet_without_head = torch.nn.Sequential(*list(mobilenet.children())[:-2])
    conv_layer1 = torch.nn.Conv2d(576, 288, kernel_size=2, stride=1, padding=0)
    conv_layer2 = torch.nn.Conv2d(288, 144, kernel_size=(3,2), stride=1, padding=1)
    conv_layer3 = torch.nn.Conv2d(144, 72, kernel_size=2, stride=1, padding=1)
    conv_layer4 = torch.nn.Conv2d(72, 2, kernel_size=2, stride=1, padding=0)
elif mobilev3_L:
    mobilenet_without_head = torch.nn.Sequential(*list(mobilenet.children())[:-2])
    conv_layer1 = torch.nn.Conv2d(960, 480, kernel_size=2, stride=1, padding=0)
    conv_layer2 = torch.nn.Conv2d(480, 120, kernel_size=(3,2), stride=1, padding=1)
    conv_layer3 = torch.nn.Conv2d(120, 60, kernel_size=2, stride=1, padding=1)
    conv_layer4 = torch.nn.Conv2d(60, 2, kernel_size=2, stride=1, padding=0)

    
head = torch.nn.Sequential(
    conv_layer1,
    torch.nn.ReLU(inplace=True),
    conv_layer2,
    torch.nn.ReLU(inplace=True),
    conv_layer3,
    torch.nn.ReLU(inplace=True),
    conv_layer4,
    torch.nn.ReLU(inplace=True),
)

model_path = './saved_models/MobileNet_Large/mobilenetV3_large_lossgeneralnorm2_nosmallcity_trial6.pth'
# Create a new model by combining the pretrained ResNet and the unflatten layer
model = torch.nn.Sequential(
    mobilenet_without_head,
    head
)
model = model.to(device)

if finetune_kitti or finetune_rgbd:
    print('Finetune!')
    model.load_state_dict(torch.load(model_path))
print(f'Model correctly loaded')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model[1][0].register_forward_hook(get_activation('conv1'))
model[1][2].register_forward_hook(get_activation('conv2'))
model[1][4].register_forward_hook(get_activation('conv3'))
model[1][6].register_forward_hook(get_activation('conv4'))

grad_clip = 1.0

resume = False
start_step = 0
start_epoch = 0
if resume:
    #TODO
    #update here start_step
    assert False, "Resume operation not yet implemented"
last_epoch = start_step if resume and start_step > 0 else -1

# Define hyperparameters for Adam optimizer
learning_rate = 0.001
#learning_rate = 0.0001
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 1e-4

# Set up the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

# Define the learning rate scheduler
# Here, we use a StepLR scheduler that multiplies the learning rate by gamma every step_size epochs
step_size = 7
gamma = 0.1
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

summary_writer = SummaryWriter(checkpoint_dir)
#logger = Logger(lr_scheduler, summary_writer, summary_freq, start_step=start_step, ttt=True)

total_steps = start_step
epoch = start_epoch
print('Start training')

min_loss_validation = 10e6
for epoch in range(start_epoch, start_epoch+total_epochs):
    print(f'Epoch {epoch+1}/{start_epoch+total_epochs}')
    model.train()
    mean_loss = 0.0
    ####################
    mean_ttt = 0.0
    mean_cok = 0.0
    mean_cko = 0.0
    ####################
    mean_loss_validation = 0.0
    batch_id = 0
    batch_id_val = 0
    for i, sample in enumerate(train_loader):
        img1, img2, gt = [x for x in sample]
        img1 = img1.to(device)
        img2 = img2.to(device)
        img1, img2 = normalize_img(img1, img2)
        input = torch.cat([img1, img2], dim=1)
        
        input = input.to(device)
        gt = gt.to(device)
        #ttt_gt = gt[:,0,:,:]
        #ttt_gt = ttt_gt.to(device)

        ttt_pred = model(input)
        if debug_dimensions:
            print(activation['conv1'].shape)
            print(activation['conv2'].shape)
            print(activation['conv3'].shape)
            print(activation['conv4'].shape)
            assert False, "Please, Check dimensions (they may be correct)"

        #loss = weighted_mse_loss(ttt_pred, gt, [1,0.5])
        loss, metrics = ttt_loss_func_general(ttt_pred, gt, norm=2,
                                      gamma=gamma,
                                      lambda_ttt=0.1, lambda_cok=0.01, lambda_cko=0.013,
                                     )

        #loss = mse_loss(ttt_pred, ttt_gt.unsqueeze(1))
        #loss = mse_loss(ttt_pred[:,0,:,:], gt[:,0,:,:])
        #loss = mse_loss(ttt_pred, gt)
        
        #print(f'\t\tBatch {i+1}/{len(train_loader)} - Loss: {loss} ({metrics})', end='\r')
        mean_loss += loss
        ############################
        mean_ttt += metrics['ttt']
        mean_cok += metrics['cok']
        mean_cko += metrics['cko']
        ############################
        
        if torch.isnan(loss):
            print(f'WARNING: Loss is nan!')
            continue
        #metrics.update({'total_loss': loss.item()})
        
        # more efficient zero_grad
        for param in model.parameters():
            param.grad = None

        loss.backward()

        ## Gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        #lr_scheduler.step()
        batch_id += 1

        #logger.push(metrics)
        #logger.add_image_summary(img1, img2, ttt_preds, gt)
    mean_loss = mean_loss/batch_id
    ############################
    m_ttt = mean_ttt/batch_id
    m_cok = mean_cok/batch_id
    m_cko = mean_cko/batch_id
    ############################
    print(f'\t\tBatch {batch_id}/{len(train_loader)} - Mean Loss: {mean_loss} (ttt: {m_ttt}, cok:{m_cok}, cko:{m_cko})', end='\r')
    #print()
    model.eval()
    #ADD eval loop
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            img1, img2, gt = [x for x in sample]
            img1 = img1.to(device)
            img2 = img2.to(device)
            img1, img2 = normalize_img(img1, img2)
            input = torch.cat([img1, img2], dim=1)
        
            input = input.to(device)
            gt = gt.to(device)
            #ttt_gt = gt[:,0,:,:]
            #ttt_gt = ttt_gt.to(device)

            ttt_pred = model(input)

            #loss_validation = weighted_mse_loss(ttt_pred, gt, [1,0.5])
            loss_validation, _ = ttt_loss_func_general(ttt_pred, gt, norm=2,
                                      gamma=gamma,
                                      lambda_ttt=0.1, lambda_cok=0.01, lambda_cko=0.013,
                                     )

            print(f'\t\tBatch {i+1}/{len(test_loader)} - Loss: {loss_validation}', end='\r')
            mean_loss_validation += loss_validation
            batch_id_val += 1
    mean_loss_validation = mean_loss_validation/batch_id_val
    ##############################
    if mean_loss_validation < min_loss_validation:
        print(f'\t\tBatch {batch_id_val}/{len(test_loader)} - Mean Loss Validation: {mean_loss_validation}   --- Best Epoch so far!', end='\r')
        min_loss_validation = mean_loss_validation
        if save:
            if mobilev3_S:
                save_path = './saved_models/MobileNet_Small/mobilenetV3_small_lossgeneralnorm2_nosmallcity_trial5.pth'
            elif mobilev3_L:
                save_path = './saved_models/MobileNet_Large/mobilenetV3_large_lossgeneralnorm2_finetunergbd_deskTest3.pth'
            else:
                assert "Error, Wrong network!"
            torch.save(model.state_dict(), save_path)
    ##############################
    else:
        print(f'\t\tBatch {batch_id_val}/{len(test_loader)} - Mean Loss Validation: {mean_loss_validation}', end='\r')
    print()
