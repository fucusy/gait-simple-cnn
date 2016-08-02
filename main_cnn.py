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

    # argument_main()
    
    # compute mean beging
    # train_data_path=config.Project.train_img_folder_path
    # test_data_path=config.Project.test_img_folder_path
    # save_file=config.Data.mean_image_file_name   

    # logging.info("train data path:%s" % train_data_path)
    # logging.info("test data path:%s" % test_data_path)

    # compute_mean_image(train_data_path, test_data_path, save_file)
    # mean image end

    train_dir = config.Project.train_img_folder_path
    test_dir = config.Project.test_img_folder_path
    cache_path = "%s/cache/" % config.Project.project_path
    feature_dir = [
                "vgg_feature_l_31_224_224",
                "vgg_feature_l_31_112_112_padding_224_224",
                ]
    feature_dir = ["%s/%s" % (cache_path, x) for x in feature_dir]

    train_data, validation_data, test_data = load_data(train_dir,test_dir,feature_dir)

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
