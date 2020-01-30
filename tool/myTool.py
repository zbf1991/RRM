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
