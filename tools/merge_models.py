import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys

fusion_prototxt = 'models/viva/ResNet-101/test.prototxt'
fusion_model = 'data/imagenet_models/ResNet101_fused.caffemodel'
even_prototxt = 'models/viva/ResNet-101_v2/test.prototxt'
even_model = 'data/imagenet_models/ResNet101_BN_SCALE_Merged.caffemodel'
odd_prototxt = 'models/viva/VGG16/test.prototxt'
odd_model = 'data/imagenet_models/VGG16.v2.caffemodel'

fusion_net = caffe.Net(fusion_prototxt, caffe.TEST)

model_list = [
    ('even', even_prototxt, even_model),
    ('odd', odd_prototxt, odd_model)
]

for prefix, model_def, model_weight in model_list:
    net = caffe.Net(model_def, model_weight, caffe.TEST)

    for layer_name, param in net.params.iteritems():
        n_params = len(param)
        try:
            for i in range(n_params):
                net.params['{}/{}'.format(prefix, layer_name)][i].data[...] = param[i].data[...]
        except Exception as e:
            print(e)

fusion_net.save(fusion_model)
