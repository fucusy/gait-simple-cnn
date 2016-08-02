__author__ = 'fucus'
import os
import skimage.io
import logging
from feature.hog import get_hog
from feature.lbp import get_lbp_his
from config import Project
import config
import pickle
from tool.keras_tool import load_train_validation_data_set, load_test_data_set
import numpy as np


cache_path = "%s/cache" % Project.project_path
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

hog_feature_cache_file_path = "%s/%s" % (cache_path, "hog_feature_cache.pickle")
lbp_feature_cache_file_path = "%s/%s" % (cache_path, "lbp_feature_cache.pickle")


def load_cache():
    # load cache
    hog_feature_cache = {}
    if os.path.exists(hog_feature_cache_file_path):
        hog_feature_file = open(hog_feature_cache_file_path, "rb")
        hog_feature_cache = pickle.load(hog_feature_file)
        hog_feature_file.close()


    lbp_feature_cache = {}
    if os.path.exists(lbp_feature_cache_file_path):
        lbp_feature_file = open(lbp_feature_cache_file_path, "rb")
        lbp_feature_cache = pickle.load(lbp_feature_file)
        lbp_feature_file.close()
    return hog_feature_cache, lbp_feature_cache

def load_feature_from_pickle(path):
    result = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_cache(hog_feature_cache, lbp_feature_cache):
    hog_feature_file = open(hog_feature_cache_file_path, "wb")
    pickle.dump(hog_feature_cache, hog_feature_file)
    hog_feature_file.close()


    lbp_feature_file = open(lbp_feature_cache_file_path, "wb")
    pickle.dump(lbp_feature_cache, lbp_feature_file)
    lbp_feature_file.close()


def image_path_to_feature(image_path_list, hog_cache, lbp_cache):
    count = 0
    features = []
    for path in image_path_list:
        if count % 1000 == 0:
            logging.info("extract %s th image feature now" % count)
        count += 1
        features.append(extract_feature(path, hog_cache, lbp_cache))
    return features


def load_train_validation_feature(img_data_path, hog_cache, lbp_cache, feature_list, limit=-1):

    train_x = []
    train_y = []

    validation_x = []
    validation_y = []

    train_data, validation_data = load_train_validation_data_set(config.Project.train_img_folder_path, to_category=False)

    train_path_list = train_data.image_path_list
    train_label_list = train_data.image_label_list

    validation_path_list = validation_data.image_path_list
    validation_label_list = validation_data.image_label_list

    if limit <= 0:
        limit = len(train_path_list)

    train_x = image_path_to_feature(train_path_list[:limit], hog_cache, lbp_cache)
    train_y = train_label_list[:limit]

    validation_x = image_path_to_feature(validation_path_list, hog_cache, lbp_cache)
    validation_y = validation_label_list

    # add feature in feature_list to train_x and validation_x
    for i in range(len(train_x)):
        img_name = os.path.basename(train_path_list[i])
        for j in range(len(feature_list)):
            train_x[i] = list(train_x[i])
            train_x[i] += list(feature_list[j][img_name][0])

    for i in range(len(validation_x)):
        img_name = os.path.basename(train_path_list[i])
        for j in range(len(feature_list)):
            validation_x[i] = list(validation_x[i])
            validation_x[i] += list(feature_list[j][img_name][0])

    return train_path_list[:limit], train_x, train_y, validation_path_list, validation_x, validation_y

def load_test_feature(img_data_path, hog_feature_cache, lbp_feature_cache, feature_list, limit=-1):

    test_img_num = 79726
    x_feature = []
    relevant_image_path_list = sorted([x for x in os.listdir("%s" % img_data_path) if x.endswith(".jpg")])
    if len(relevant_image_path_list) != test_img_num:
        logging.warning("the test images number:%d is not equal to %d, it's incorrect"
                        % (len(relevant_image_path_list), test_img_num))
    else:
        logging.info("the test images number:%d is equal to %d, it's correct"
                        % (len(relevant_image_path_list), test_img_num))
    logging.info("start to load feature from test image")


    test_data = load_test_data_set(config.Project.test_img_folder_path)
    relevant_image_path_list = [os.path.basename(x) for x in test_data.image_path_list]

    count = 0
    for img in relevant_image_path_list:
        if count >= limit > 0:
            break
        if count % 1000 == 0:
            logging.info("extract %s th image feature now" % count)
        count += 1
        img_path = "%s/%s" % (img_data_path, img)
        x_feature.append(extract_feature(img_path, hog_feature_cache, lbp_feature_cache))

    logging.info("load feature from test image end")

    # add feature in feature_list to train_x and validation_x
    for i in range(len(x_feature)):
        img_name = relevant_image_path_list[i]
        for j in range(len(feature_list)):
            x_feature[i] = list(x_feature[i])
            x_feature[i] += list(feature_list[j][img_name][0])

    return relevant_image_path_list[:count], x_feature[:count]

def extract_feature(img_path, hog_feature_cache, lbp_feature_cache):
    img_name = img_path.split("/")[-1]
    img = skimage.io.imread(img_path)
    feature = []
    if img_name in hog_feature_cache:
        hog_feature = hog_feature_cache[img_name]
    else:
        hog_feature = get_hog(img)
        hog_feature_cache[img_name] = hog_feature


    if img_name in lbp_feature_cache:
        lbp_feature = lbp_feature_cache[img_name]
    else:
        lbp_feature = get_lbp_his(img)
        lbp_feature_cache[img_name] = lbp_feature

    feature += list(hog_feature)
    feature += list(lbp_feature)

    return feature
