__author__ = 'fucus'
import os
from tool.keras_tool import load_data
import logging
import sys
import datetime
import config
from config import Project
from feature.utility import load_train_validation_feature
from feature.utility import load_test_feature
from tool.file import generate_result_file
from feature.utility import load_cache
from feature.utility import load_feature_from_pickle
from feature.utility import save_cache
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
import numpy as np


cache_path = "%s/cache" % Project.project_path

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    start_time = datetime.datetime.now()
    logging.info('start program---------------------')
    logging.info("loading feature cache now")

    train_img_folder = "/home/chenqiang/kaggle_driver_data/imgs/train"
    test_img_folder = "/home/chenqiang/kaggle_driver_data/imgs/test"

    feature_dir_list = [
                "vgg_feature_l_31_crop_336_336_224_224",
                "hand_feature_crop_336_336_224_224_get_hog",
                "vgg_feature_l_31_crop_336_336_112_112_padding_224_224",
                "vgg_feature_l_36_crop_336_336_224_224"
                ]
    feature_dir_list = ["%s/%s" % (cache_path, x) for x in feature_dir_list]

    for feature_dir in feature_dir_list:
        if os.path.exists(feature_dir):
            logging.info("load feature from %s" % os.path.basename(feature_dir))

    train_data, validation_data, test_data = load_data(train_img_folder, test_img_folder)
    
    train_y = train_data.get_image_label(to_cate=False)
    validation_y = validation_data.get_image_label(to_cate=False)


    logging.info("train_y shape %s" % str(train_y.shape)) 
    logging.info("validation_y shape %s" % str(validation_y.shape)) 

    feature_list = [None, None]
    for j, dataset in enumerate([train_data, validation_data]):
        for i, path in enumerate(dataset.image_path_list):
            x = np.array([])
            img_base_name = os.path.basename(path)
            for feature_dir in feature_dir_list:
                feature_file_name = "%s/%s.npy" % (feature_dir, img_base_name)
                if os.path.exists(feature_file_name) and \
                        os.path.isfile(feature_file_name):
                    feature = np.load(feature_file_name)
                    x = np.append(x, feature, axis=0)
            if feature_list[j] is None:
                feature_list[j] = x
            else:
                feature_list[j] = np.vstack((feature_list[j], x))
            if i % 100 == 0:
                logging.info("x.shape = %s" % x.shape)
                logging.info("load feature of %dth %s at dataset %d" % (i, path, j))

    train_data_feature = feature_list[0] 
    validation_data_feature = feature_list[1]

    logging.info("load feature done")
    logging.info("train_data_feature shape %s" % str(train_data_feature.shape)) 
    logging.info("validation_data_feature shape %s" % str(validation_data_feature.shape)) 


    logging.info("start to train the model")

    Project.predict_model.fit(x_train=train_data_feature, y_train=train_y
                              , x_validation=validation_data_feature, y_validation=validation_y)

    logging.info("train the model done")
    logging.info("start to do validation")
    validation_result = Project.predict_model.predict(validation_data_feature) 
    report = classification_report(validation_result, validation_y)
    logging.info("the validation report:\n %s" % report)

    validation_pro = Project.predict_model.predict_proba(validation_data_feature) 
    logloss_val =  log_loss(validation_y, validation_pro)

    logging.info("validation logloss is %.3f" % logloss_val)
    logging.info("done validation")

    logging.info("start predict test data")
    predict_result = None
    for i, path in enumerate(test_data.image_path_list):
        x = np.array([])
        img_base_name = os.path.basename(path)
        for feature_dir in feature_dir_list:
            feature_file_name = "%s/%s.npy" % (feature_dir, img_base_name)
            if os.path.exists(feature_file_name) and \
                    os.path.isfile(feature_file_name):
                feature = np.load(feature_file_name)
                x = np.append(x, feature, axis=0)
        predict = Project.predict_model.predict_proba(x)

        if predict_result is None:
            predict_result = predict
        else:
            predict_result = np.vstack((predict_result, predict))
        if i % 100 == 0:
            logging.info("test image feature of %dth %s" % (i, path))
    logging.info("predict test data done")

    logging.info("start to generate the final file used to submit")
    generate_result_file(test_data.image_path_list, predict_result)
    logging.info("generated the final file used to submit")

    end_time = datetime.datetime.now()
    logging.info('total running time: %.2f second' % (end_time - start_time).seconds)
    logging.info('end program---------------------')
