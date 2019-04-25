import numpy as np

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('snlive_roi_predictor')
class SNLIVERoiPredictor(Predictor):

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        label_id = np.argmax(output_dict['label_logits'], axis=-1)

        label_token = self._model.vocab.get_token_from_index(label_id, 'labels')
        output_dict['prediction'] = label_token
        return output_dict

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        flickr_id = json_dict['Flikr30kID'][:-4]

        image, boxes = self._dataset_reader.load_image_and_boxes(flickr_id)

        meta_dict = {
            'pair_id': json_dict['pairID'],
            'caption_id': json_dict['captionID'],
            'flickr_id': flickr_id
        }

        return self._dataset_reader.text_to_instance(
            premise_img=image,
            boxes=boxes,
            hypothesis=json_dict['sentence2'],
            meta_dict=meta_dict
        )
