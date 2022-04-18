import torch
import math
import torchvision
import cv2
import random
import numpy as np
from ocr.warp_mls import WarpMLS


def get_train_transforms(height, width, prob):
    transforms = torchvision.transforms.Compose([
        UseWithProb(RandomGaussianBlur(max_ksize=7), prob=prob),
        #UseWithProb(RandomRotate(2), prob),
#         UseWithProb(RandomCrop(rnd_crop_min=0.85), prob),
        RescalePaddingImage(height, width),
        # UseWithProb(Perspective(), prob),
#         UseWithProb(Stretch(2, 5), prob),
#         UseWithProb(Distortion(2, 4), prob),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


def get_val_transforms(height, width):
    transforms = torchvision.transforms.Compose([
        RescalePaddingImage(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms

def get_test_transforms(height, width):
    transforms = torchvision.transforms.Compose([
        RescalePaddingImage(height, width),
        # Denoising(),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms

class Thinning:
    def __init__(self, k, iterations=1):
        self.k = k
        self.iterations = iterations
        
    def __call__(self, img):
        kernel = np.ones((self.k, self.k), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=self.iterations)
        return erosion
    
    
class AddBorder:
    def __init__(self, sizes, color=[0, 0, 0]):
        assert len(sizes) == 4, 'Sizes should be len 4.'
        assert len(color) == 3, 'Color should be len 3.'
        self.sizes = sizes
        self.color = color
        
    def __call__(self, img):
        a, b, c, d = self.sizes
        return cv2.copyMakeBorder(img, a, b, c, d, cv2.BORDER_CONSTANT, None, self.color)


class Skeletonization:
    def __call__(self, img):
        thinned = cv2.ximgproc.thinning(img)
        return thinned
    

class Binarization:
    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return 255 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)

    
class Denoising:
    def __call__(self, img):
        img = cv2.fastNlMeansDenoising(img, None, 3, 7, 11)
        return img
    
    
class UpScale:
    def __init__(self, k=2):
        self.k = k

    def __call__(self, img):
        return cv2.resize(img, (img.shape[1] * self.k, img.shape[0] * self.k))

    
class Dilate():
    def __init__(self, k, iterations=1):
        self.k = k
        self.iterations = 1
            
    def __call__(self, img):
        kernel = np.ones((self.k, self.k), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=self.iterations)
        return dilation

    
class GammaBrightness:
    def __init__(self, alpha=1, beta=0):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, img):
        new_image = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
        return new_image


class Stretch:
    def __init__(self, min_k=3, max_k=3):
        self.min_k = min_k
        self.max_k = max_k

    def __call__(self, src):
        segment = random.randint(self.min_k, self.max_k)
        img_h, img_w = src.shape[:2]
        cut = img_w // segment
        thresh = cut * 4 // 5
        src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        dst_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        half_thresh = thresh * 0.5
        for cut_idx in np.arange(1, segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])
        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()
        return dst


class Distortion:
    def __init__(self, min_k=3, max_k=3):
        self.min_k = min_k
        self.max_k = max_k
    
    def __call__(self, src):
        segment = random.randint(self.min_k, self.max_k)
        img_h, img_w = src.shape[:2]
        cut = img_w // segment
        thresh = cut // 3
        src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        dst_pts = [[np.random.randint(thresh), np.random.randint(thresh)],
                   [img_w - np.random.randint(thresh), np.random.randint(thresh)],
                   [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)],
                   [np.random.randint(thresh), img_h - np.random.randint(thresh)]]
        half_thresh = thresh * 0.5
        for cut_idx in np.arange(1, segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            np.random.randint(thresh) - half_thresh])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            img_h + np.random.randint(thresh) - half_thresh])

        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()
        return dst


class Perspective:
    def __call__(self, src):
        img_h, img_w = src.shape[:2]
        thresh = img_h // 2
        src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        dst_pts = [[0, np.random.randint(thresh)],
                   [img_w, np.random.randint(thresh)],
                   [img_w, img_h - np.random.randint(thresh)],
                   [0, img_h - np.random.randint(thresh)]]
        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()

        return dst


class RescalePaddingImage:
    def __init__(self, output_height, output_width):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image):
        h, w = image.shape[:2]
        # print(image.shape)
        new_width = int(w * (self.output_height / h))
        new_width = min(new_width, self.output_width)
        image = cv2.resize(image, (new_width, self.output_height),
                           interpolation=cv2.INTER_LINEAR)
        if new_width < self.output_width:
            image = np.pad(
                image, ((0, 0), (0, self.output_width - new_width), (0, 0)),
                'constant', constant_values=0)
        return image


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        return np.moveaxis(image, 0, -1)


class UseWithProb:
    def __init__(self, transform, prob=0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class RandomGaussianBlur:
    def __init__(self, max_ksize=5, sigma_x=20):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, image):
        kernal_size = tuple(2 * np.random.randint(0, self.max_ksize, 2) + 1)
        blured_image = cv2.GaussianBlur(image, kernal_size, self.sigma_x)
        return blured_image


def img_crop(img, bbox):
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def random_crop(img, size):
    tw = size[0]
    th = size[1]
    h, w = img.shape[:2]
    if ((w - tw) > 0) and ((h - th) > 0):
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
    else:
        x1 = 0
        y1 = 0
    img_return = img_crop(img, (x1, y1, x1 + tw, y1 + th))
    return img_return, x1, y1


class RandomCrop:
    def __init__(self, rnd_crop_min, rnd_crop_max=1):
        self.factor_max = rnd_crop_max
        self.factor_min = rnd_crop_min

    def __call__(self, img):
        factor = random.uniform(self.factor_min, self.factor_max)
        size = (
            int(img.shape[1]*factor),
            int(img.shape[0]*factor)
        )
        img, x1, y1 = random_crop(img, size)
        return img


def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


class RandomRotate:
    def __init__(self, max_ang=0):
        self.max_ang = max_ang

    def __call__(self, img):
        h, w, _ = img.shape

        ang = np.random.uniform(-self.max_ang, self.max_ang)
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1)
        img = cv2.warpAffine(img, M, (w, h))

        w_cropped, h_cropped = largest_rotated_rect(w, h, math.radians(ang))
        img = crop_around_center(img, w_cropped, h_cropped)
        return img


class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = get_val_transforms(height, width)

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor
    
class Denoising:
    def __call__(self, img):
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

