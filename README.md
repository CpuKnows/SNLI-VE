# SNLI-VE

Experiments with multi-modal entailment using an early fusion model and an attention model over words and image objects.

SNLI-VE corpus compiled by Xie et al. (2018)

## Data
* Download [Flickr30k images](http://shannon.cs.illinois.edu/DenotationGraph/)  
* Download [ELMo weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
* Download [ELMo options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* Download SNLI-VE dataset. See [SNLI-VE repo](https://github.com/necla-ml/SNLI-VE) for more information.
  * [SNLI-VE train split](https://drive.google.com/file/d/1jQElLXUA5ps3OuiSMlJdKTRIMwZ_cJ2e/view?usp=sharing)
  * [SNLI-VE dev split](https://drive.google.com/file/d/1M6uSoJ4rXXsygReioHSJg2Xgip2NLpJW/view?usp=sharing)
  * [SNLI-VE test split](https://drive.google.com/file/d/1_n4g8sbw_P6KBayvJ9B8KklZDFr0uPsQ/view?usp=sharing)

## Setup

For full setup instructions see [INSTALL.md](./INSTALL.md)

## SNLI-VE Models

### Fasttext hypothesis only baseline
Run `scripts/create_fasttext_datasets.py` to generate files for fasttext. 
Run `scripts/create_snli_hard.py` to create hard dataset splits.   

Train fasttext model and make predictions:
```
fasttext supervised -input fasttext_train.txt -ouput fasttext_hyp_only -wordNgrams 2
fasttext predict fasttext_hyp_only.bin fasttext_<split>.txt 1 > prediction_<split>.txt 
```

### Detectron bounding boxes for ROI Attention models
Run inference for bounding boxes:
```
DETECTRON=/path/to/detectron
SNLIVE=/path/to/SNLI-VE
python $DETECTRON/tools/infer_snlive.py \
    --cfg $DETECTRON/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml \
    --output-dir $SNLIVE/data/detectron \
    --output-ext json \
    --image-ext jpg \
    --wts $DETECTRON/weights/e2e_mask_rcnn_R-50-FPN_2x_model.pkl \
    $SNLIVE/data/flickr30k-images
```
The custom detection script can be found in `scripts/infer_snlive.py`

### SNLI-VE training and inference
Create smaller data subsets for training runs `scripts/subset_snli_ve_data.py`

Training:  
```
allennlp train experiments/<EXPERIMENT_NAME>.json \
    --serialization-dir models/<EXPERIMENT_NAME> \
    --include-package snli_ve
```

Evaluation for fusion models:
```
allennlp predict \
    --output-file data/predictions/<OUTPUT>.json \
    --silent \
    --cuda-device -1 \
    --predictor snlive_fusion_predictor \
    --include-package snli_ve \
    models/<EXPERIMENT_NAME>/model.tar.gz \
    data/snli_ve_<SPLIT>.jsonl
```

Evaluation for ROI Attention models:
```
allennlp predict \
    --output-file data/predictions/<OUTPUT>.json \
    --silent \
    --cuda-device -1 \
    --predictor snlive_roi_predictor \
    --include-package snli_ve \
    models/<EXPERIMENT_NAME>/model.tar.gz \
    data/snli_ve_<SPLIT>.jsonl
```

## Results
__Total dataset__
<table>
<tr>
    <th></th>
    <th colspan="4">Validation set</th>
    <th colspan="4">Test set</th>
</tr>
<tr>
    <th>Model</th>
    <th>Overall</th>
    <th>Entailed</th>
    <th>Neutral</th>
    <th>Contradict</th>
    <th>Overall</th>
    <th>Entailed</th>
    <th>Neutral</th>
    <th>Contradict</th>
</tr>
<tr>
    <td>Hypothesis only</td>
    <td>64.50</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>64.20</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>Early fusion</td>
    <td>62.86</td>
    <td>68.97</td>
    <td>64.61</td>
    <td>54.96</td>
    <td>63.09</td>
    <td>69.31</td>
    <td>65.38</td>
    <td>54.56</td>
</tr>
<tr>
    <td>Early fusion with ELMo</td>
    <td>67.05</td>
    <td>70.15</td>
    <td>62.23</td>
    <td>68.78</td>
    <td>67.07</td>
    <td>69.36</td>
    <td>62.63</td>
    <td>69.23</td>
</tr>
<tr>
    <td>ROI Attention</td>
    <td>63.34</td>
    <td>70.46</td>
    <td>64.85</td>
    <td>54.69</td>
    <td>63.47</td>
    <td>69.98</td>
    <td>65.64</td>
    <td>54.76</td>
</tr>
</table>

__Hard dataset__
<table>
<tr>
    <th></th>
    <th colspan="4">Validation set</th>
    <th colspan="4">Test set</th>
</tr>
<tr>
    <th>Model</th>
    <th>Overall</th>
    <th>Entailed</th>
    <th>Neutral</th>
    <th>Contradict</th>
    <th>Overall</th>
    <th>Entailed</th>
    <th>Neutral</th>
    <th>Contradict</th>
</tr>
<tr>
    <td>Hypothesis only</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>Early fusion</td>
    <td>21.97</td>
    <td>26.36</td>
    <td>27.45</td>
    <td>12.24</td>
    <td>21.89</td>
    <td>25.50</td>
    <td>27.75</td>
    <td>12.47</td>
</tr>
<tr>
    <td>Early fusion with ELMo</td>
    <td>32.19</td>
    <td>33.42</td>
    <td>27.19</td>
    <td>36.48</td>
    <td>32.09</td>
    <td>31.16</td>
    <td>27.40</td>
    <td>37.86</td>
</tr>
<tr>
    <td>ROI Attention</td>
    <td>19.49</td>
    <td>25.83</td>
    <td>23.65</td>
    <td>09.49</td>
    <td>19.70</td>
    <td>24.99</td>
    <td>23.79</td>
    <td>10.62</td>
</tr>
</table>

## Citations

Ning Xie, Farley Lai, Derek Doran, and Asim Kadav. "Visual Entailment Task for Visually-Grounded Language Learning." arXiv preprint arXiv:1811.10582 (2018).