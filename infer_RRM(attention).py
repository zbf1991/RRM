import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import imageio
import importlib
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score, labels=21)

    return crf_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='./netWeights/RRM(attention)_final.pth', type=str)
    parser.add_argument("--network", default="network.RRM(attention)", type=str)
    parser.add_argument("--out_cam_pred", default='./output/result/no_crf', type=str)
    parser.add_argument("--out_la_crf", default='./output/result/crf', type=str)
    parser.add_argument("--LISTpath", default="./voc12/val(id).txt", type=str)
    parser.add_argument("--IMpath", default="/home/zbf/dataset/VOCdevkit/VOC2012/JPEGImages", type=str)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()
    im_path = args.IMpath
    img_list = open(args.LISTpath).readlines()
    pred_softmax = torch.nn.Softmax(dim=0)
    for i in img_list:

        img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg'))
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
        img_original = img_temp.astype(np.uint8)
        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

        input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()

        output = model(input)[2]
        output = F.interpolate(output, (img_temp.shape[0], img_temp.shape[1]),mode='bilinear',align_corners=False)

        output = torch.squeeze(output)
        pred_prob = pred_softmax(output)
        output = torch.argmax(output,dim=0).cpu().numpy()

        save_path = os.path.join(args.out_cam_pred,i[:-1] + '.png')
        cv2.imwrite(save_path,output.astype(np.uint8))
        print(i)

        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(pred_prob, img_original)

            crf_img = np.argmax(crf_la, 0)

            imageio.imsave(os.path.join(args.out_la_crf, i[:-1] + '.png'), crf_img.astype(np.uint8))

