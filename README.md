# SNLI-VE

## Data
* Download [Flickr30k images](http://shannon.cs.illinois.edu/DenotationGraph/)  
* Download [ELMo weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
* Download [ELMo options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* Download SNLI-VE dataset. See [SNLI-VE repo](https://github.com/necla-ml/SNLI-VE) for more information.
  * [SNLI-VE train split](https://drive.google.com/file/d/1jQElLXUA5ps3OuiSMlJdKTRIMwZ_cJ2e/view?usp=sharing)
  * [SNLI-VE dev split](https://drive.google.com/file/d/1M6uSoJ4rXXsygReioHSJg2Xgip2NLpJW/view?usp=sharing)
  * [SNLI-VE test split](https://drive.google.com/file/d/1_n4g8sbw_P6KBayvJ9B8KklZDFr0uPsQ/view?usp=sharing)

## Setup

### Fasttext
Fasttext (Joulin et al., 2017) is used as a baseline and for generating the _SNLI-VE hard_ dataset given only the text hypothesis as input.

See [fasttext.cc](https://fasttext.cc/docs/en/support.html) for setup information.

Run `scripts/create_fasttext_datasets.py` to generate files for fasttext. Run `scripts/create_snli_hard.py` to create hard dataset splits.   

Train fasttext model and make predictions:
```
fasttext supervised -input fasttext_train.txt -ouput fasttext_hyp_only -wordNgrams 2
fasttext predict fasttext_hyp_only.bin fasttext_<split>.txt 1 > prediction_<split>.txt 
```

### Facebook Detectron
Detectron is used to get image bounding boxes of objects using a pretrained ResNet-50 model. 

See [Detectron repo](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md) for setup information.

The end-to-end Mask R-CNN baseline with the ResNet-50 architecture is used for bounding box detection. 
[Model weights](https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) 
are available from the Detectron model zoo (Girshick et al., 2018).

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
The detection script can be found in `scripts/infer_snlive.py`

### SNLI-VE Project
Requires Linux

Image features: https://keras.io/applications/#resnet

```
conda create --name snlive python=3.6
source activate snlive

conda install pytorch cudatoolkit=9.0 -c pytorch
# For ROIAlign layer
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d

pip install allennlp
python -m spacy download en_core_web_sm

# this one is optional but it should help make things faster
pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

Create smaller data subsets for training run `scripts/subset_snli_ve_data.py`

To perform training:  
```
allennlp train experiments/${EXPERIMENT_NAME}.json --serialization-dir models/${EXPERIMENT_NAME} --include-package snli_ve
```

To perform evaluation: