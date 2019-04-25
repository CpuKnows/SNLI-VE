# Setup

## SNLI-VE Project
Requires Linux. ROI Attention model requires a GPU.

```
conda create --name snlive python=3.6
source activate snlive

conda install h5py
conda install pytorch cudatoolkit=9.0 -c pytorch
# For ROIAlign layer
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d

pip install allennlp
python -m spacy download en_core_web_sm

# this one is optional but it should help make things faster
pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

## Fasttext
Fasttext (Joulin et al., 2016) is used as a baseline and for generating the _SNLI-VE hard_ dataset given only the text hypothesis as input.

See [fasttext.cc](https://fasttext.cc/docs/en/support.html) for setup information.

## Facebook Detectron
Detectron is used to get image bounding boxes of objects using a pretrained ResNet-50 model. 

See [Detectron repo](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md) for setup information.

The end-to-end Mask R-CNN baseline with the ResNet-50 architecture is used for bounding box detection. 
[Model weights](https://dl.fbaipublicfiles.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) 
are available from the Detectron model zoo (Girshick et al., 2018).

## Citations

Ross Girshick, Ilija Radosavovic, Georgia Gkioxari, Piotr Dollár, and Kaiming He. "Detectron." https://github.com/facebookresearch/detectron. (2018).

Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. "Bag of tricks for efficient text classification." arXiv preprint arXiv:1607.01759 (2016).