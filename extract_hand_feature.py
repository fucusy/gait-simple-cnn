import sys
import os
import config
from preprocess.resize import resize_image_main
from preprocess.argument import argument_main
from tool.data_tools import compute_mean_image
from tool.model_tools import KerasModel
from tool.keras_tool import load_data
from model.cnn_model import VGG_16_freeze
import numpy as np
import logging
import feature.hog as feature_factory

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    train_img_folder = [
    "/home/chenqiang/kaggle_driver_data/imgs/train_crop_336_336_224_224"]

    # required, this path contain imgs to be tested
    test_img_folder = "/home/chenqiang/kaggle_driver_data/imgs/test_crop_336_336_224_224"

    train_data, validation_data, test_data = load_data(train_img_folder, test_img_folder)

    logging.info("train data image count %s" % train_data.count())
    logging.info("validation data image count %s" % validation_data.count())

 
    # get feature function
    feature_func_name = 'get_hog'
    extract_function = getattr(feature_factory, feature_func_name)

    # feature saving path
    base_name = os.path.basename(test_img_folder)[5:]
    base_path = config.Project.project_path
    numpy_file_dir = "%s/cache/hand_feature_%s_%s"\
                    % (base_path, base_name, feature_func_name)

    if not os.path.exists(numpy_file_dir):
        os.makedirs(numpy_file_dir)

    fragment_size = 64
    count = 0
    for dataset in [train_data, validation_data, test_data]:
        while dataset.have_next():
            image_list, path_list = dataset.next_fragment(fragment_size, preprocess_fuc=None)
            for i in range(len(image_list)):
                count += 1
                result = extract_function(image_list[i])
                with open("%s/%s.npy" % (numpy_file_dir, path_list[i])
                        , 'wb') as f:
                    np.save(f, result)
                if count % 100 == 0:
                    logging.info("extract %d image feature" % count)
                    logging.info("feature shape: %s" % result.shape)
    logging.info("save feature to %s" % numpy_file_dir)
