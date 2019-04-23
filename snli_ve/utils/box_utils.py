import numpy as np
from PIL.Image import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional


def load_image(img_fn: str) -> Image:
    """[H, W, 3] numpy array"""
    return default_loader(img_fn)


def resize_image(image: Image,
                 min_height=224,
                 min_width=224) -> (Image, (int, int)):
    """
    Pad image to at least min_height , min_width.
    Odd padding will round down on left and top, round up on right and bottom.

    :param image:
    :param min_height:
    :param min_width:
    :return: padded image,
             original image [x1, y1, x2, y2]
             padding [left, top, right, bottom]
    """
    w, h = image.size
    h_pad, w_pad = 0, 0

    if h < min_height:
        h_pad = min_height - h

    if w < min_width:
        w_pad = min_width - w

    padding = (w_pad // 2, h_pad // 2, np.ceil(w_pad / 2).astype(np.int), np.ceil(h_pad / 2).astype(np.int))
    image = functional.pad(image, padding)
    window = [padding[0], padding[1], padding[0] + w, padding[1] + h]

    return image, window, padding


def to_tensor_and_normalize(image: Image):
    """Normalization for ImageNET"""
    return functional.normalize(functional.to_tensor(image), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
