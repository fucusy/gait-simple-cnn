#!/usr/bin/python2.7

import sys
import config
from preprocess.resize import resize_image_main
from preprocess.argument import argument_main
from tool.data_tools import compute_mean_image

from tool.model_tools import KerasModel
from tool.keras_tool import load_data
import model.cnn_model as model_factory
import numpy as np


import logging

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_img_dirs = ["/home/chenqiang/data/gait-simple-cnn-data/nm-data_extract_210_70"]
    test_img_dirs = []
    train_data, validation_data, test_data = load_data(train_img_dirs,test_img_dirs)

    logging.info("train data image count %s" % train_data.count())
    logging.info("validation data image count %s" % validation_data.count())

    model_func = getattr(model_factory, config.CNN.model_name)
    
    weights_path = config.CNN.keras_train_weight
    lr = config.CNN.lr

    # calculate the input dim
    input = 0
    for d in feature_dir:
        feature = np.load("%s/img_0.jpg.npy" % d)
        input += feature.shape[0]

    cnn_model = model_func(lr, weights_path,input)

    model = KerasModel(cnn_model=cnn_model)
    model.train_model(train_data, validation_data, save_best=True)
    model.predict_model(test_data)
    model.save_prediction()
