import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.layers import ROIAlign


def _load_resnet():
    # https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    # Change to match caffe resnet
    backbone = resnet.resnet50(pretrained=True)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)
    return backbone


class Flattener(nn.Module):
    def __init__(self):
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleDetector(nn.Module):
    def __init__(self, final_dim=1024):
        super(SimpleDetector, self).__init__()
        self.final_dim = final_dim
        self.downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flattener(),
            nn.Dropout(p=0.1),
            nn.Linear(2048, self.final_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_feats):
        return self.downsample(img_feats)


class ROIDetector(nn.Module):
    """Uses precomputed bounding boxes and masks"""

    def __init__(self, final_dim):
        super(ROIDetector, self).__init__()
        self.final_dim = final_dim

        backbone = _load_resnet()
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            #backbone.layer4
        )

        self.roi_align = ROIAlign((7, 7), spatial_scale=1 / 16, sampling_ratio=0)

        self.after_roi_align = nn.Sequential(
            backbone.layer4,
            nn.AvgPool2d(7, stride=1),
            Flattener()
        )

        self.obj_downsample = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2048, final_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self,
                images: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                #classes: torch.Tensor = None,
                #segms: torch.Tensor = None
                ):
        """

        :param images: [batch_size, 3, im_height, im_width
        :param boxes: [batch_size, max_num_objects, 4]
        :param box_mask: [batch_size, max_num_objects]
        :return: [batch_size, max_num_objects, dim]
        """

        img_feats = self.backbone(images)

        box_inds = box_mask.nonzero()  # [num nonzero, 2] (x, y) indices
        assert box_inds.shape[0] > 0  # at least 1 masked index
        rois = torch.cat((
            box_inds[:, 0, None].type(boxes.dtype),  # [x * y, 1]
            boxes[box_inds[:, 0], box_inds[:, 1]]  # boxes[x * y, 4]
        ), 1)  # [x * y, 1] + [x * y, 4] -> [x * y, 5]

        roi_align_res = self.roi_align(img_feats, rois)

        post_roi_align = self.after_roi_align(roi_align_res)

        #obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]

        roi_aligned_feats = self.obj_downsample(post_roi_align)

        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())
        return {
            'obj_reps_raw': post_roi_align,
            'obj_reps': obj_reps,
            #'obj_labels': obj_labels
        }


def pad_sequence(sequence, lengths):
    """
    :param sequence: [\sum b, .....] sequence
    :param lengths: [b1, b2, b3...] that sum to \sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output
