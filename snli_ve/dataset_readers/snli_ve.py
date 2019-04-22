from typing import Dict, Iterator
import json
import logging
import h5py
import numpy as np

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ArrayField, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("snlive")
class SNLIVEDatasetReader(DatasetReader):
    def __init__(self,
                 img_h5fn: str,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.img_h5fn = img_h5fn
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, premise_img, hypothesis, label=None) -> Instance:
        fields: Dict[str, Field] = {}
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields['premise_img'] = ArrayField(premise_img, dtype=np.float32)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
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
                with h5py.File(self.img_h5fn, 'r') as h5:
                    img_feat = np.array(h5[flickr_id], dtype=np.float32)
                # TF tensor format is channel last while PyTorch is channel first
                img_feat = img_feat.transpose((0, 3, 1, 2))
                img_feat = np.squeeze(img_feat, axis=0)

                yield self.text_to_instance(img_feat, sample['sentence2'], sample['gold_label'])
