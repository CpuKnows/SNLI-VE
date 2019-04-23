from typing import Dict, Iterator, Optional
import json
import logging
import os
import numpy as np

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ArrayField, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

from snli_ve.utils.mask_utils import segms_to_mask
from snli_ve.utils.box_utils import load_image, resize_image, to_tensor_and_normalize

logger = logging.getLogger(__name__)


@DatasetReader.register("snlive_detectron")
class SNLIVERoiDatasetReader(DatasetReader):
    def __init__(self,
                 img_dir: str,
                 metadata_dir: str,
                 add_image_as_box: bool = True,
                 max_boxes: Optional[int] = None,
                 min_box_prob: Optional[float] = 0.7,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        """

        :param img_dir: contains images
        :param metadata_dir: contains detectron output
        :param add_image_as_box: add original image as a box detection
        :param max_boxes:
        :param min_box_prob:
        :param tokenizer:
        :param token_indexers:
        :param lazy:
        """
        super().__init__(lazy)
        self.img_dir = img_dir
        self.metadata_dir = metadata_dir
        self.add_image_as_box = add_image_as_box
        self.max_boxes = max_boxes
        self.min_box_prob = min_box_prob
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    #    def _read_boxes(self, metadata_fn):
    #        with open(os.path.join(self.metadata_dir, metadata_fn), 'r') as f:
    #            metadata = json.load(f)
    #            masks = segms_to_mask(metadata['segms'])
    #            metadata['segms'] = masks
    #            return metadata

    def text_to_instance(self, premise_img, boxes, hypothesis, meta_dict, label=None) -> Instance:
        fields: Dict[str, Field] = {}
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields['premise_img'] = ArrayField(premise_img, dtype=np.float32)
        fields['boxes'] = ArrayField(boxes, padding_value=-1)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        fields['metadata'] = MetadataField(meta_dict)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as f:
            for line in f:
                sample = json.loads(line)

                if sample['gold_label'] == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue

                flickr_id = sample['Flikr30kID'][:-4]
                image = load_image(os.path.join(self.img_dir, flickr_id + '.jpg'))
                image, window, img_padding = resize_image(image)
                image = to_tensor_and_normalize(image)

                # Load boxes
                with open(os.path.join(self.metadata_dir, flickr_id + '.json'), 'r') as mf:
                    metadata = json.load(mf)

                # TODO: if using class labels and segms these must end up in the same order
                boxes = np.array(metadata['boxes'])
                # Keep highest probability box detections
                if self.min_box_prob is not None:
                    to_keep = boxes[:, 4] > self.min_box_prob
                    boxes = boxes[to_keep]
                # Upper bound on boxes to use
                if self.max_boxes is not None and boxes.shape[0] > self.max_boxes:
                    sort_idx = np.flipud(boxes[:, 4].argsort())
                    boxes = boxes[sort_idx][:self.max_boxes]

                # remove confidence
                boxes = boxes[:, :-1]
                # adjust box coordinates by left and top padding
                # img_padding [left, top, right, bottom]
                boxes[:, :2] += np.array(img_padding[:2])[None]

                if self.add_image_as_box:
                    boxes = np.row_stack((window, boxes))

                if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
                    import ipdb
                    ipdb.set_trace()
                assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
                assert np.all((boxes[:, 2] <= image.shape[2]))
                assert np.all((boxes[:, 3] <= image.shape[1]))

                meta_dict = {
                    'pair_id': sample['pairID'],
                    'caption_id': sample['captionID'],
                    'flickr_id': flickr_id
                }

                yield self.text_to_instance(image, boxes, sample['sentence2'], meta_dict, sample['gold_label'])
