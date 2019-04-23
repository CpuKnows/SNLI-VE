import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_final_encoder_states, get_text_field_mask, masked_softmax, weighted_sum, \
    replace_masked_values
from allennlp.nn import InitializerApplicator

from snli_ve.utils.detector_util import ROIDetector

logger = logging.getLogger(__name__)


@Model.register("ROIAttention")
class ROIAttention(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 regularizer: Optional[RegularizerApplicator] = None,
                 detector_final_dim: int = 512,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        """

        :param vocab:
        :param text_field_embedder:
        :param encoder:
        :param output_feedforward:
        :param regularizer:
        :param detector_final_dim:
        :param dropout:
        :param initializer:
        """
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self.detector = ROIDetector(detector_final_dim)

        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim
        )

        self._output_feedforward = output_feedforward

        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                premise_img: torch.Tensor,
                boxes: torch.Tensor,
                hypothesis: Dict[str, torch.Tensor],
                metadata: List[Dict],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """

        :param premise_img:
        :param boxes:
        :param hypothesis:
        :param metadata:
        :param label:
        :return:
        """
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self.rnn_input_dropout:
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        box_mask = torch.all(boxes >= 0, -1)
        obj_reps = self.detector(premise_img, boxes, box_mask)

        hyp2obj_sim = self.obj_attention(encoded_hypothesis, obj_reps)
        hyp2obj_att_weights = masked_softmax(hyp2obj_sim, box_mask[:, None, None])
        attended_o = torch.einsum('bnao,bod->bnad', hyp2obj_att_weights, obj_reps['obj_reps'])

        label_logits = self._output_feedforward(attended_o)
        label_probs = nn.functional.softmax(label_logits, dim=-1)

        output_dict = {
            "label_logits": label_logits,
            "label_probs": label_probs
        }

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
