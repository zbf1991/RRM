import random
import numpy as np
from PIL import Image, ImageOps
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from torchvision import transforms

class Compose(transforms.Compose):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_dict):
        img = img_dict[0]
        ori_img = img_dict[1]
        croppings = img_dict[2]
        for t in self.transforms:
            img, ori_img, croppings = t(img, ori_img, croppings)
        return img, ori_img, croppings


class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img, ori_img, croppings):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]
        croppings = np.ones_like(imgarr)
        return proc_img, imgarr, croppings


class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, ori_img, croppings):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        ori_img = img
        croppings = croppings
        return img, ori_img, croppings



class RandomResizeLong():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img):
        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        return img

class Scale_im():
    def __init__(self,scale):
        self.Scale = scale

    def __call__(self, img):

        new_dims = (int(img.shape[0] * self.scale), int(img.shape[1] * self.scale))
        return cv2.resize(img, new_dims)


class Crop():
    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        if self.cropsize < h or self.cropsize < w:
            print("crop size must larger than image size")

        container = 254 * np.ones((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[0:h, 0:w] = imgarr[0:h, 0:w]

        return container
class Resize():
    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        container = cv2.resize(imgarr, (self.cropsize, self.cropsize))

        return container



class FixScaleCropImage(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img

class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr, ori_img, croppings):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

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

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        ori_contrainer = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        croppings = np.zeros((self.cropsize, self.cropsize), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]
        ori_contrainer[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            ori_img[img_top:img_top+ch, img_left:img_left+cw]
        croppings[cont_top:cont_top+ch, cont_left:cont_left+cw] = 1

        return container, ori_contrainer, croppings

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def crop_with_box(img, box):
    if len(img.shape) == 3:
        img_cont = np.zeros((max(box[1]-box[0], box[4]-box[5]), max(box[3]-box[2], box[7]-box[6]), img.shape[-1]), dtype=img.dtype)
    else:
        img_cont = np.zeros((max(box[1] - box[0], box[4] - box[5]), max(box[3] - box[2], box[7] - box[6])), dtype=img.dtype)
    img_cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return img_cont


def random_crop(images, cropsize, fills):
    if isinstance(images[0], Image.Image):
        imgsize = images[0].size[::-1]
    else:
        imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, fills):

        if isinstance(img, Image.Image):
            img = img.crop((box[6], box[4], box[7], box[5]))
            cont = Image.new(img.mode, (cropsize, cropsize))
            cont.paste(img, (box[2], box[0]))
            new_images.append(cont)

        else:
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            new_images.append(cont)

    return new_images


class AvgPool2d():

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, img):
        import skimage.measure

        return skimage.measure.block_reduce(img, (self.ksize, self.ksize, 1), np.mean)


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(img, ori_img, croppings):
    return np.transpose(img, (2, 0, 1)), ori_img, croppings

def HWC_to_CHW_VAL(img):
    return np.transpose(img, (2, 0, 1))


class RescaleNearest():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, npimg):
        import cv2
        return cv2.resize(npimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)


class RandomScaleCrop(object):
    def __init__(self, base_size=513, crop_size=513, fill=254):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, img, label):
        img = img
        mask = label
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, mask






def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img_c), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def crf_inference_inf(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83/scale_factor, srgb=5, rgbim=np.copy(img_c), compat=4)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)