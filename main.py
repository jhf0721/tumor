import os
import numpy as np
import json, time
from functools import partial
import nibabel as nb
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast #native AMP
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.transforms.transform import MapTransform
import sys
from os import environ

from monai.inferers import sliding_window_inference
# from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import load_decathlon_datalist
from monai.transforms import AsDiscrete,Activations,Compose

from monai import transforms, data
from monai_trainer import AMDistributedSampler, run_training
from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR
from networks.unetr import UNETR
from networks.swin3d_unetr import SwinUNETR
from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='5fold cross val')

parser.add_argument('--syn',default=True,action='store_true')
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--logdir', default="runs/synt.no_pretrain.swin_unetrv2_tiny_no_examine")
parser.add_argument('--save_checkpoint',default=True, action='store_true')
parser.add_argument('--max_epochs', default=4000, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--optim_lr', default=4e-4, type=float)

parser.add_argument('--optim_name', default='adamw', type=str)
parser.add_argument('--reg_weight', default=1e-5, type=float)

parser.add_argument('--noamp', default=True,action='store_true') #experimental
parser.add_argument('--val_every', default=200, type=int)
parser.add_argument('--val_overlap', default=0.5, type=float)
parser.add_argument('--cache_rate', default=1.0, type=float)

parser.add_argument('--distributed',action='store_true') #distributed multi gpu
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23457', type=str,  help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=1, type=int)


parser.add_argument('--model_name', default='swinunetr', type=str)
parser.add_argument('--swin_type', default='tiny', type=str)
parser.add_argument('--tumor_type', default='tumor', type=str)
parser.add_argument('--organ_type', default='liver', type=str)
parser.add_argument('--organ_model', default='liver', type=str)
parser.add_argument('--diff_model', default=None, type=str)
parser.add_argument('--ddim_ts', default=50, type=int)
parser.add_argument('--fg_thresh', default=10, type=int)
parser.add_argument('--healthy_num', default=400, type=int)
parser.add_argument('--healthy_seed', default=0, type=int)
#segmentation flex params
parser.add_argument('--seg_block', default='basic_pre', type=str)
parser.add_argument('--seg_num_blocks', default = '1,2,2,4', type=str)
parser.add_argument('--seg_base_filters', default=16, type=int)
parser.add_argument('--seg_relu', default='relu', type=str)
parser.add_argument('--seg_lastnorm_init_zero', action='store_true')

parser.add_argument('--seg_mode', default=1, type=int)

parser.add_argument('--seg_use_se', action='store_true')
parser.add_argument('--seg_norm_name', default='instancenorm', type=str)
parser.add_argument('--seg_noskip', action='store_true')
parser.add_argument('--seg_aug_mode', default=0, type=int)
parser.add_argument('--seg_aug_noflip', action='store_true')

parser.add_argument('--seg_norm_mode', default=0, type=int)
parser.add_argument('--seg_crop_mode', default=0, type=int)

#unetr params
parser.add_argument('--pos_embedd', default='conv', type=str)
parser.add_argument('--norm_name', default='instance', type=str)
parser.add_argument('--num_steps', default=40000, type=int)
parser.add_argument('--eval_num', default=500, type=int)
parser.add_argument('--warmup_steps', default=500, type=int)
parser.add_argument('--num_heads', default=16, type=int)
parser.add_argument('--mlp_dim', default=3072, type=int)
parser.add_argument('--hidden_size', default=768, type=int)
parser.add_argument('--in_channels', default=1, type=int)
parser.add_argument('--out_channels', default=3, type=int)
parser.add_argument('--num_classes', default=3, type=int)
parser.add_argument('--res_block', action='store_true')
parser.add_argument('--conv_block', action='store_true')
parser.add_argument('--roi_x', default=96, type=int)
parser.add_argument('--roi_y', default=96, type=int)
parser.add_argument('--roi_z', default=64, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--decay', default=1e-5, type=float)
parser.add_argument('--lrdecay', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--amp_scale', action='store_true')
parser.add_argument('--opt_level', default='O2', type=str)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lrschedule', default='warmup_cosine', type=str)
parser.add_argument('--warmup_epochs', default=100, type=int)
parser.add_argument('--resume_ckpt', action='store_true')
parser.add_argument('--pretrained_dir', default=None, type=str)
parser.add_argument('--data_root', default=None, type=str)
parser.add_argument('--healthy_data_root', default=None, type=str)
parser.add_argument('--datafold_dir', default=None, type=str)
parser.add_argument('--cache_num', default=200, type=int)
parser.add_argument('--train_dir', default=None, type=str)
parser.add_argument('--val_dir', default=None, type=str)
parser.add_argument('--use_pretrained', action='store_true')
parser.add_argument('--json_dir', default="datafolds/gen_no_examine.json", type=str)
class RandCropByPosNegLabeld_select(transforms.RandCropByPosNegLabeld):
    def __init__(self, keys, label_key, spatial_size, 
                 pos=1.0, neg=1.0, num_samples=1, 
                 image_key=None, image_threshold=0.0, allow_missing_keys=True,
                   fg_thresh=0):
        super().__init__(keys=keys, label_key=label_key, spatial_size=spatial_size, 
                 pos=pos, neg=neg, num_samples=num_samples, 
                 image_key=image_key, image_threshold=image_threshold, allow_missing_keys=allow_missing_keys)
        self.fg_thresh = fg_thresh

    def R2voxel(self,R):
        return (4/3*np.pi)*(R)**(3)

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']
        d.pop('name')
        if 'kidney_label' in data_name or 'liver_label' in data_name or 'pancreas_label' in data_name:
            flag=0
            while 1:
                flag+=1
                d_crop = super().__call__(d)
                pixel_num = (d_crop[0]['label']>0).sum()
                # print(pixel_num)
                if pixel_num > self.R2voxel(self.fg_thresh):
                    break
                if flag>5 and pixel_num > self.R2voxel(max(self.fg_thresh-5, 5)):
                    break
                if flag>10 and pixel_num > self.R2voxel(max(self.fg_thresh-10, 5)):
                    break
                if flag>15 and pixel_num > self.R2voxel(max(self.fg_thresh-15, 5)):
                    break
                if flag>20 and pixel_num > self.R2voxel(max(self.fg_thresh-20, 5)):
                    break
                if flag>25 and pixel_num > self.R2voxel(max(self.fg_thresh-25, 5)):
                    break
                if flag>30:
                    break
        else:
            d_crop = super().__call__(d)
        d_crop[0]['name'] = data_name

        return d_crop

class LoadImage_train(MapTransform):
    def __init__(self,organ_type):
        self.reader1 = transforms.LoadImaged(keys=["image", "label"])
        self.organ_type = organ_type

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']

        if (not 'kidney_label' in data_name) and self.organ_type == 'kidney':
            d = self.reader1.__call__(d)
            d['label'][d['label']==3] = 1
        elif ('kidney_label' in data_name) and self.organ_type == 'kidney':
            d = self.reader1.__call__(d)
            d['label'][d['label']>0] = 1
            
        else :
            d = self.reader1.__call__(d)

        return d
    
class LoadImage_val(transforms.LoadImaged):
    def __init__(self, keys, *args,**kwargs, ):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']

        d = super().__call__(d)
        d['label'][d['label']==3] = 1

        return d
    
def _get_transform(args):

    train_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    return train_transform, val_transform

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)

        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))

    else:
        # Simply call main_worker function
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True) #in new Pytorch/python labda functions fail to pickle with spawn
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)

    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu) #use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    print(args.rank, ' gpu', args.gpu)
    if args.rank==0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    roi_size = [args.roi_x, args.roi_y, args.roi_z]
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    
    data_root = args.data_root
    healthy_data_root = args.healthy_data_root
    datafold_dir = args.datafold_dir
    fold = args.fold
    tumor_type = args.tumor_type
    organ_type = args.organ_type
    if organ_type == 'liver':
        args.fg_thresh = 30
    elif organ_type == 'pancreas':
        args.fg_thresh = 15
    elif organ_type == 'kidney':
        args.fg_thresh = 25
    train_transform, val_transform = _get_transform(args)
    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
        print('Logging to', args.logdir)

    ## NETWORK
    if (args.model_name is None) or args.model_name == 'unet':
        from monai.networks.nets import UNet 
        model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
        model_dict = torch.load(os.path.join(args.logdir, 'model.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('Use pretrained weights')
        
    elif args.model_name == 'swinunetr':
        
        if args.swin_type == 'tiny':
            feature_size=12
        elif args.swin_type == 'small':
            feature_size=24
        elif args.swin_type == 'base':
            feature_size=48

        model = SwinUNETR_v2(in_channels=1,
                          out_channels=3,
                          img_size=(96, 96, 96),
                          feature_size=feature_size,
                          patch_size=2,
                          depths=[2, 2, 2, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=[7, 7, 7])
        model_dict = torch.load(os.path.join(args.logdir, 'model.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('Use pretrained weights')
        if args.use_pretrained:
            pretrained_add = 'model_swinvit.pt'
            model.load_from(weights=torch.load(pretrained_add))
            print('Use pretrained ViT weights from: {}'.format(pretrained_add))
    elif args.model_name == 'nnunet':
        from monai.networks.nets import DynUNet
        from dynunet_pipeline.create_network import get_kernels_strides
        from dynunet_pipeline.task_params import deep_supr_num
        task_id = 'custom'
        kernels, strides = get_kernels_strides(task_id)
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=deep_supr_num[task_id],
        )
        model_dict = torch.load(os.path.join(args.logdir, 'nnunet_synt_liver_tumors.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('Use pretrained weights')
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))
        
    if args.resume_ckpt:
        model_dict = torch.load(args.pretrained_dir)
        model.load_state_dict(model_dict['state_dict'])
        print('Use pretrained weights')


    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    post_label = AsDiscrete(to_onehot=True, n_classes=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.num_classes)
    val_channel_names=['val_liver_dice', 'val_tumor_dice']

    print('Crop size', roi_size)
    data_dir = args.train_dir
    val_data_dir = args.val_dir
    datalist_json = args.json_dir
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=val_data_dir)

    new_val_files =[]
    for item in val_files:
        new_item = {}
        new_item['image'] = item['image'].replace('.npy', '.gz')
        new_item['label'] = item['label'].replace('.npy', '.gz')
        new_val_files.append(new_item)

    val_shape_dict = {}

    for d in val_files:
        imagepath = d["image"]
        imagename = imagepath.split('/')[-1]
        imgnb = nb.load(imagepath)
        val_shape_dict[imagename] = [imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]]
    print('Totoal number of validation: {}'.format(len(val_shape_dict)))
    train_ds = data.SmartCacheDataset(
        data=datalist,
        transform=train_transform,
        cache_num=args.cache_num,
        cache_rate=1.0,
        # num_init_workers=args.workers // 2,
        num_init_workers=1,
        num_replace_workers=4
    )

    train_sampler = AMDistributedSampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=1,
                                   sampler=train_sampler, pin_memory=True)
    val_ds = data.Dataset(data=new_val_files, transform=val_transform)

    val_sampler = AMDistributedSampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, sampler=val_sampler,
                                 pin_memory=True)


    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,  overlap=args.val_overlap, mode='gaussian')

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)



    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))


    model.cuda(args.gpu)

    

    
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name=='batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu) #??

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)


    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=0.99, nesterov=True, weight_decay=args.reg_weight) #momentum 0.99, nestorov=True, following nnUnet
    else:
        raise ValueError('Unsupported optim_name' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )


    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)

    else:
        scheduler = None



    accuracy = run_training(model=model,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             optimizer=optimizer,
                             loss_func=dice_loss,
                             args=args,
                             model_inferer=model_inferer,
                             scheduler=scheduler,
                             start_epoch=start_epoch,
                             val_channel_names=val_channel_names,
                             val_shape_dict=val_shape_dict,
                             post_label=post_label,
                             post_pred=post_pred, val_acc_max = best_acc)

    return accuracy


if __name__ == '__main__':
    main()
