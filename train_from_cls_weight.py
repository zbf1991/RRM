import os
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from tool import pyutils, torchutils
import argparse
import importlib
import torch.nn.functional as F
from DenseEnergyLoss import DenseEnergyLoss

import tool.myTool as mytool
from tool.myTool import compute_joint_loss, compute_seg_label, compute_cam_up



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU_id')

    parser.add_argument("--LISTpath", default="voc12/train_aug(id).txt", type=str)
    parser.add_argument("--IMpath", default="/home/zbf/dataset/VOCdevkit/VOC2012/JPEGImages", type=str)
    parser.add_argument("--SAVEpath", default="./output/model_weights", type=str)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_step", default=20000, type=int)
    parser.add_argument("--network", default="network.RRM", type=str)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--weights",default='./netWeights/res38_cls.pth', type=str)

    parser.add_argument("--session_name", default="RRM_", type=str)
    parser.add_argument("--crop_size", default=321, type=int)
    parser.add_argument("--class_numbers", default=20, type=int)

    parser.add_argument('--crf_la_value', type=int, default=4)
    parser.add_argument('--crf_ha_value', type=int, default=32)

    parser.add_argument('--densecrfloss', type=float, default=1e-7,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=100,
                        help='DenseCRF sigma_xy')

    args = parser.parse_args()

    gpu_id = args.gpu_id

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    save_path = os.path.join(args.SAVEpath,args.session_name)
    print("dloss weight", args.densecrfloss)
    critersion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean').cuda()
    DenseEnergyLosslayer = DenseEnergyLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb,
                                     sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)

    model = getattr(importlib.import_module(args.network), 'SegNet')()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    max_step = args.max_step

    batch_size = args.batch_size
    img_list = mytool.read_file(args.LISTpath)

    data_list = []
    for i in range(int(max_step//100)):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer_cls([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    data_gen = mytool.chunker(data_list, batch_size)

    for iter in range(max_step + 1):
        chunk = data_gen.__next__()
        img_list = chunk
        images, ori_images, label, croppings = mytool.get_data_from_chunk_v2(chunk,args)
        b, _, w, h = ori_images.shape
        c = args.class_numbers
        label = label.cuda(non_blocking=True)

        x_f, cam, seg = model(images, require_seg = True, require_mcam = True)
        cam_up = compute_cam_up(cam, label, w, h, b)
        seg_label = np.zeros((b,w,h))
        for i in range(b):
            cam_up_single = cam_up[i]
            cam_label = label[i].cpu().numpy()
            ori_img = ori_images[i].transpose(1,2,0).astype(np.uint8)
            norm_cam = cam_up_single/(np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)
            seg_label[i] = compute_seg_label(ori_img, cam_label, norm_cam)

        closs = F.multilabel_soft_margin_loss(x_f, label)

        celoss, dloss = compute_joint_loss(ori_images, seg[0], seg_label, croppings, critersion, DenseEnergyLosslayer)
        loss = closs + celoss + dloss
        print('closs: %.4f'% closs.item(),'celoss: %.4f'%celoss.item(), 'dloss: %.4f'%dloss.item())

        avg_meter.add({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (optimizer.global_step - 1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                  'Loss:%.4f' % (avg_meter.pop('loss')),
                  'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                  'Fin:%s' % (timer.str_est_finish()),
                  'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            if (optimizer.global_step - 1) % 2000 == 0 and optimizer.global_step > 10000:
                torch.save(model.module.state_dict(), save_path + '%d.pth' % (optimizer.global_step - 1))

    torch.save(model.module.state_dict(), args.session_name + 'final.pth')

