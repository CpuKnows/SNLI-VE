from typing import Dict, List
import numpy as np
from pycocotools.mask import decode


def segms_to_mask(segms: [Dict, List]) -> np.ndarray:
    """
    Convert segment(s) from COCO RLE format to masks

    :param segms: dict('counts': RLE format, 'size': [height, width]),
    :return: array of masks
    """
    return decode(segms)
