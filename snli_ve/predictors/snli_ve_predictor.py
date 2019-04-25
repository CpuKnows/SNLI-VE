import h5py
import numpy as np

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('snlive_fusion_predictor')
class SNLIVEFusionPredictor(Predictor):

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        label_id = np.argmax(output_dict['label_logits'], axis=-1)

        label_token = self._model.vocab.get_token_from_index(label_id, 'labels')
        output_dict['prediction'] = label_token
        return output_dict

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        flickr_id = json_dict['Flikr30kID'][:-4]

        with h5py.File(self._dataset_reader.img_h5fn, 'r') as h5:
            img_feat = np.array(h5[flickr_id], dtype=np.float32)
        # TF tensor format is channel last while PyTorch is channel first
        img_feat = img_feat.transpose((0, 3, 1, 2))
        img_feat = np.squeeze(img_feat, axis=0)

        return self._dataset_reader.text_to_instance(
            premise_img=img_feat,
            hypothesis=json_dict['sentence2']
        )
