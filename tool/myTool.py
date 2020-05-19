import numpy as np
import tool.imutils as imutils
import torch
import torch.nn.functional as F
import cv2
import random
import os

def _crf_with_alpha(ori_img,cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(ori_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = np.zeros([21, bg_score.shape[1], bg_score.shape[2]])
    n_crf_al[0, :, :] = crf_score[0, :, :]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def compute_seg_label(ori_img, cam_label, norm_cam):
    cam_label = cam_label.astype(np.uint8)

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(20):
        if cam_label[i] > 1e-5:
            cam_dict[i] = norm_cam[i]
            cam_np[i] = norm_cam[i]

    bg_score = np.power(1 - np.max(cam_np, 0), 32)
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))
    _, bg_w, bg_h = bg_score.shape

    cam_img = np.argmax(cam_all, 0)

    crf_la = _crf_with_alpha(ori_img, cam_dict, 4)
    crf_ha = _crf_with_alpha(ori_img, cam_dict, 32)
    crf_la_label = np.argmax(crf_la, 0)
    crf_ha_label = np.argmax(crf_ha, 0)
    crf_label = crf_la_label.copy()
    crf_label[crf_la_label == 0] = 255

    single_img_classes = np.unique(crf_la_label)
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool)
    for class_i in single_img_classes:
        if class_i != 0:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            cam_class_order = cam_class[cam_class > 0.1]
            cam_class_order = np.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * 0.6)
            confidence_value = cam_class_order[confidence_pos]
            class_sure_region = (cam_class > confidence_value)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
        else:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            class_sure_region = (cam_class > 0.8)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)

    cam_not_sure_region = ~cam_sure_region

    crf_label[crf_ha_label == 0] = 0
    crf_label_np = np.concatenate([np.expand_dims(crf_ha[0, :, :], axis=0), crf_la[1:, :, :]])
    crf_not_sure_region = np.max(crf_label_np, 0) < 0.8
    not_sure_region = np.logical_or(crf_not_sure_region, cam_not_sure_region)

    crf_label[not_sure_region] = 255

    return crf_label


def compute_joint_loss(ori_img, seg, seg_label, croppings, critersion, DenseEnergyLosslayer):
    seg_label = np.expand_dims(seg_label,axis=1)
    seg_label = torch.from_numpy(seg_label)

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    pred = F.interpolate(seg,(w,h),mode="bilinear",align_corners=False)
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred)
    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2,0,1))
    dloss = DenseEnergyLosslayer(ori_img,pred_probs,croppings, seg_label)
    dloss = dloss.cuda()

    seg_label_tensor = seg_label.long().cuda()

    seg_label_copy = torch.squeeze(seg_label_tensor.clone())
    bg_label = seg_label_copy.clone()
    fg_label = seg_label_copy.clone()
    bg_label[seg_label_copy != 0] = 255
    fg_label[seg_label_copy == 0] = 255
    bg_celoss = critersion(pred, bg_label.long().cuda())

    fg_celoss = critersion(pred, fg_label.long().cuda())

    celoss = bg_celoss + fg_celoss

    return celoss, dloss


def compute_cam_up(cam, label, w, h, b):
    cam_up = F.interpolate(cam, (w, h), mode='bilinear', align_corners=False)
    cam_up = cam_up * label.clone().view(b, 20, 1, 1)
    cam_up = cam_up.cpu().data.numpy()
    return cam_up


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def resize_label_batch(label, size):
    label_resized = np.zeros((size, size, 1, label.shape[3]))
    interp = torch.nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = torch.autograd.Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>21] = 255
    return label_resized


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def RandomCrop(imgarr, cropsize):

    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    img_container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)

    cropping =  np.zeros((cropsize, cropsize), np.bool)


    img_container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1

    return img_container, cropping

def get_data_from_chunk_v2(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)
    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    for i, piece in enumerate(chunk):
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        img_temp = scale_im(img_temp, scale)
        img_temp = flip(img_temp, flip_p)
        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping = RandomCrop(img_temp, dim)
        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = cropping.astype(np.float32)

        images[:, :, :, i] = img_temp

    images = images.transpose((3, 2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    return images, ori_images, labels, croppings


def compute_cos(fts1, fts2):
    fts1_norm2 = torch.norm(fts1, 2, 1).view(-1, 1)
    fts2_norm2 = torch.norm(fts2, 2, 1).view(-1, 1)

    fts_cos = torch.div(torch.mm(fts1, fts2.t()), torch.mm(fts1_norm2, fts2_norm2.t()) + 1e-7)

    return fts_cos


def compute_dis_no_batch(seg, seg_feature):
    seg = torch.argmax(seg, dim=1, keepdim=True).view(seg.shape[0],1, -1)
    seg_no_batch = seg.permute(0,2,1).clone().view(-1,1)

    bg_label = torch.zeros_like(seg).float()

    bg_label[seg == 0] = 1
    bg_num = torch.sum(bg_label) + 1e-7

    seg_feature = seg_feature.view(seg_feature.shape[0], seg_feature.shape[1], -1)

    seg_feature_no_batch = seg_feature.permute(0, 2, 1).clone()
    seg_feature_no_batch = seg_feature_no_batch.view(-1, seg_feature.shape[1])

    seg_feature_bg = seg_feature * bg_label
    bg_num_batch = torch.sum(bg_label, dim=2)+1e-7
    seg_feature_bg_center = torch.sum(seg_feature_bg, dim=2) / bg_num_batch
    pixel_dis = 0

    bg_center_num = 0
    for batch_i in range(seg_feature.shape[0]):
        bg_num_batch_i = bg_num_batch[batch_i]
        bg_pixel_dis = 1-compute_cos(seg_feature[batch_i].transpose(1,0), seg_feature_bg_center[batch_i].unsqueeze(dim=0))
        if bg_num_batch_i>=1:
            pixel_dis += (torch.sum(bg_pixel_dis * bg_label[batch_i].transpose(1,0), dim=0)/ bg_num_batch_i)
        else:
            pixel_dis += 2*torch.ones([1]).cuda()

        bg_center_num+=1

    fg_center_num=0
    seg_feature_fg_center = torch.zeros([1, 1024])
    batch_num = 0
    for i in range(1, 21):
        class_label = torch.zeros_like(seg_no_batch).float()
        class_label[seg_no_batch == i] = 1
        class_num = torch.sum(class_label) + 1e-7
        batch_num += class_num
        if class_num < 1:
            continue
        else:
            seg_feature_class = seg_feature_no_batch * class_label
            seg_feature_class_center = torch.sum(seg_feature_class, dim=0, keepdim=True) / class_num
            fg_pixel_dis = 1-compute_cos(seg_feature_no_batch, seg_feature_class_center)
            pixel_dis += (torch.sum(fg_pixel_dis*class_label,dim=0)/ class_num)
            fg_center_num += 1
            if fg_center_num == 1:
                seg_feature_fg_center = seg_feature_class_center
            else:
                seg_feature_fg_center = torch.cat([seg_feature_fg_center, seg_feature_class_center], dim=0)

    pixel_dis = pixel_dis / (fg_center_num+bg_center_num)

    if batch_num >= 1 and torch.sum(bg_num) >= 1:

        fg_fg_cos = 1 + compute_cos(seg_feature_fg_center, seg_feature_fg_center)
        fg_bg_cos = 1 + compute_cos(seg_feature_fg_center, seg_feature_bg_center)

        fg_fg_cos = fg_fg_cos - torch.diag(torch.diag(fg_fg_cos))
        if fg_fg_cos.shape[0]>1:
            fg_fg_loss = torch.sum(fg_fg_cos) / (fg_fg_cos.shape[0] * (fg_fg_cos.shape[1] - 1))

        else:
            fg_fg_loss = torch.zeros([1]).cuda()
        fg_bg_loss = torch.sum(fg_bg_cos) / (fg_bg_cos.shape[0] * fg_bg_cos.shape[1])
        dis_loss = 0.5 * fg_fg_loss.cuda() + 0.5 * fg_bg_loss.cuda()

    elif torch.sum(bg_num) < 1:
        fg_norm2 = torch.norm(seg_feature_fg_center, 2, 1).view(-1, 1)

        fg_fg_cos = 1 + torch.div(torch.mm(seg_feature_fg_center, seg_feature_fg_center.t()),
                                  torch.mm(fg_norm2, fg_norm2.t()) + 1e-7)

        fg_fg_cos = fg_fg_cos - torch.diag(torch.diag(fg_fg_cos))

        if fg_fg_cos.shape[0]>1:
            fg_fg_loss = torch.sum(fg_fg_cos) / (fg_fg_cos.shape[0] * (fg_fg_cos.shape[1] - 1))

        else:
            fg_fg_loss = torch.zeros([1]).cuda()

        dis_loss = 0.5 * fg_fg_loss + 1

    else:
        dis_loss = torch.zeros([1]).cuda()

    return dis_loss.cuda()+pixel_dis.cuda()