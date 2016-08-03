# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:58:44 2016
    
@author: WuPeng
"""

import sys
# add project root to python lib search path
sys.path.append("../")

import config
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D

from keras.models import model_from_json
from keras.optimizers import SGD
import logging
import h5py
import os
from tool.keras_tool import load_model

logger = logging.getLogger('cnn_model')


def simple_cnn_vgg_like(lr=1e-3, weights_path=None):
    img_rows, img_cols = 210, 70
    # standard VGG16 network architecture
    
    structure_path = "%s/cache/simple_cnn_vgg_like.json" % config.Project.project_path
    if weights_path is not None and os.path.exists(weights_path) \
        and os.path.exists(structure_path):

        logger.debug("load weigth from fine-tuning weight %s" % weights_path)
        model = model_from_json(open(structure_path).read())
        model.load_weights(weights_path)
    else:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))

        # replace more fc layer
        model.add(Dense(124, activation='softmax'))

        # load the weights
        logger.debug('Model loaded.')

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])

    return model

def simple_cnn_for_test(lr=1e-3, weights_path=None):
    img_rows, img_cols = 210, 70
    if weights_path is not None and os.path.exists(weights_path):
        logging.debug("load weigth from fine-tuning weight %s" % weights_path)
        model = load_model(weights_path)
    else:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(1, img_rows, img_cols)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        # replace more fc layer
        model.add(Dense(124, activation='softmax'))

        # load the weights
        logging.debug('Model loaded.')

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])
    return model



def VGG_16_replace_fc(lr=1e-3, weights_path=None):
    img_rows, img_cols, color_type = 224, 224, 3
    # standard VGG16 network architecture

    structure_path = "%s/cache/VGG_16_replace_fc.json" % config.Project.project_path
    if weights_path is not None and os.path.exists(weights_path) \
        and os.path.exists(structure_path):

        logger.debug("load weigth from fine-tuning weight %s" % weights_path)
        model = model_from_json(open(structure_path).read())
        model.load_weights(weights_path)

    else:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                     img_rows, img_cols)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())

        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # load the weights
        
        f = h5py.File(config.Project.vgg_weight_file_path)
        
        # load weight except fully connected layers
        for k in range(31):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            logging.info("set %d layer weight to %s" % (k, weights))
        f.close()
        logging.debug("load weigth from vgg weight")
        logging.debug('Model loaded.')

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

def VGG_16_add_layer(lr=1e-3, weights_path=None):
    img_rows, img_cols, color_type = 224, 224, 3
    # standard VGG16 network architecture
    
    if weights_path is not None and os.path.exists(weights_path):
        logging.debug("load weigth from fine-tuning weight %s" % weights_path)
        model = load_model(weights_path)
    else:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                     img_rows, img_cols)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))

        # replace more fc layer
        model.add(Dense(10, activation='softmax'))

        # load the weights
        
        f = h5py.File(config.Project.vgg_weight_file_path)
        
        # we don't look at the last (fully-connected) layers in the savefile
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            logging.info("set %d layer weight to %s" % (k, weights))
        f.close()
        logging.debug("load weigth from vgg weight")
        logging.debug('Model loaded.')

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])

    return model

def VGG_16():
    img_rows, img_cols, color_type = 224, 224, 3
    # standard VGG16 network architecture
    
    vgg_weight = config.Project.vgg_weight_file_path

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # load the weights
    model.load_weights(config.Project.vgg_weight_file_path)
    logging.debug("load weigth from vgg weight")
    logging.debug('Model loaded.')

    lr = 1e-3
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])

    return model


def VGG_16_freeze(lr=1e-3, weights_path=None):
    # standard VGG16 network architecture
    img_rows, img_cols, color_type = 224, 224, 3
    vgg_weight = config.Project.vgg_weight_file_path

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols), trainable=False))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))

    # layer 9, start from 0
    model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))

    # layer 19 
    model.add(ZeroPadding2D((1, 1), trainable=False))

    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1), trainable=False))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1, 1), trainable=False))

    # layer 29, start from 0
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    model.add(Flatten(trainable=False))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    weights_path_exist = False
    if weights_path is not None and os.path.exists(weights_path):
        weights_path_exist = True

    if weights_path is None or not weights_path_exist:
        logging.debug("load weigth from vgg weight")
        model.load_weights(config.Project.vgg_weight_file_path)

    # load model weights
    if weights_path is not None and weights_path_exist:
        logging.debug("load weigth from fine-tuning weight %s" % weights_path)
        model.load_weights(weights_path)

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])
    logger.info("init with model lr:%s" % lr)

    return model


def VGG_16_add_layer_freeze(lr=1e-3, weights_path=None):
    img_rows, img_cols, color_type = 224, 224, 3
    # standard VGG16 network architecture
    
    structure_path = "%s/cache/VGG_16_add_layer_freeze.json" % config.Project.project_path
    if weights_path is not None and os.path.exists(weights_path) \
        and os.path.exists(structure_path):

        logger.debug("load weigth from fine-tuning weight %s" % weights_path)
        model = model_from_json(open(structure_path).read())
        model.load_weights(weights_path)
    else:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                     img_rows, img_cols), trainable=False))
        model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(ZeroPadding2D((1, 1), trainable=False))
        model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))
    
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', trainable=False))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', trainable=False))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu', trainable=False))

        model.add(Dense(1000, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # load the weights
        
        f = h5py.File(config.Project.vgg_weight_file_path)
        
        # we don't look at the last (fully-connected) layers in the savefile
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            logging.info("set %d layer weight to %s" % (k, weights))
        f.close()
        logging.debug("load weigth from vgg weight")
        logging.debug('Model loaded.')

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])

    return model

def vgg_std16_model(img_rows, img_cols, color_type=1, model_weights_file='', continueFile='', optimizer='sgd'):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    if not continueFile:
        print('loading orininal vgg weights')
        model.load_weights(model_weights_file)

