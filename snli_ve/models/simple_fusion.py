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

from snli_ve.utils.detector_util import SimpleDetector

logger = logging.getLogger(__name__)


@Model.register("SimpleFusion")
class SimpleFusion(Model):
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
        self.detector = SimpleDetector(detector_final_dim)

        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward

        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                premise_img: torch.LongTensor,
                hypothesis: Dict[str, torch.Tensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """

        :param premise_img:
        :param hypothesis:
        :param label:
        :return:
        """
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        if self.rnn_input_dropout:
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        hypothesis_hidden_state = get_final_encoder_states(
            encoded_hypothesis,
            hypothesis_mask,
            self._encoder.is_bidirectional()
        )

        img_feats = self.detector(premise_img)

        fused_features = torch.cat((img_feats, hypothesis_hidden_state), dim=-1)

        label_logits = self._output_feedforward(fused_features)
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
