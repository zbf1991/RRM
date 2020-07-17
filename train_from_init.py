import os
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils
import argparse
import importlib
import torch.nn.functional as F
from DenseEnergyLoss import DenseEnergyLoss
from tool.myTool import compute_seg_label, compute_joint_loss, compute_cam_up
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def validate(model, data_loader):
    print('\nvalidating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack[1]
            label = pack[2].cuda(non_blocking=True)

            x = model(img, require_seg=False, require_mcam=False)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss:', val_loss_meter.pop('loss'))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=24, type=int)
    parser.add_argument("--network", default="network.RRM", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", default='./netWeights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="RRM_", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--class_numbers", default=20, type=int)
    parser.add_argument("--voc12_root", default='/home/zbf/dataset/VOCdevkit/VOC2012', type=str)

    parser.add_argument('--densecrfloss', type=float, default=1e-7,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=100.0,
                        help='DenseCRF sigma_xy')

    args = parser.parse_args()

    save_path = os.path.join("../psa_zbf/output/model_weights",
                             args.session_name)

    print("dloss weight", args.densecrfloss)
    critersion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean').cuda()
    DenseEnergyLosslayer = DenseEnergyLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb,
                                     sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)

    model = getattr(importlib.import_module(args.network), 'SegNet')()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(256, 512),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray]),
                        transform2=
                        imutils.Compose([imutils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    val_dataset = voc12.data.VOC12ClsDatasetVAL(args.val_list, voc12_root=args.voc12_root,
                                             transform=transforms.Compose([
                                                 np.asarray,
                                                 model.normalize,
                                                 imutils.CenterCrop(500),
                                                 imutils.HWC_to_CHW_VAL,
                                                 torch.from_numpy
                                             ]))
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        assert args.network == "network.RRM"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):
            images = pack[1]
            ori_images = pack[3].numpy().transpose(0,3,1,2)
            label = pack[2].cuda(non_blocking=True)
            croppings = pack[4].numpy().transpose(1,2,0)

            b, _, w, h = ori_images.shape
            c = args.class_numbers
            label = label.cuda(non_blocking=True)
            if (optimizer.global_step - 1) < 0.5*optimizer.max_step:
                x_f = model(images, require_seg=False, require_mcam=False)
                closs = F.multilabel_soft_margin_loss(x_f, label)
                loss = closs
                print('closs', closs.data)
            else:
                x_f, cam, seg = model(images, require_seg=True, require_mcam=True)
                cam_up = compute_cam_up(cam, label, w, h, b)
                seg_label = np.zeros((b, w, h))
                cam_weight = np.zeros((b, w, h))
                for i in range(b):
                    cam_up_single = cam_up[i]
                    cam_label = label[i].cpu().numpy()
                    ori_img = ori_images[i].transpose(1, 2, 0).astype(np.uint8)
                    norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)
                    seg_label[i] = compute_seg_label(ori_img, cam_label, norm_cam)

                closs = F.multilabel_soft_margin_loss(x_f, label)

                celoss, dloss = compute_joint_loss(ori_images, seg[0], seg_label, croppings, critersion, DenseEnergyLosslayer)

                loss = closs + celoss + dloss
                print('closs: %.4f'% closs.item(),'celoss: %.4f'%celoss.item(), 'dloss: %.4f'%dloss.item())

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                if (optimizer.global_step - 1) % 2000 == 0 and optimizer.global_step > 10000:
                    torch.save(model.module.state_dict(), save_path + '%d.pth' % (optimizer.global_step - 1))

        else:
            # validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + 'final.pth')
