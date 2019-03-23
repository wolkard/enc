#-*- coding:utf-8 –*-
import pickle
import cv2
import numpy as np
import math,random,os,re,time
from keras.models import Model, model_from_json
from keras import backend as K

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )

import sys
import json
sys.path.append("/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master")

import os
from src.utils import preprocess
import src.config as cf
from src.data_generator import (
    TextSequenceGenerator,
    decode_predict_ctc,
    labels_to_text,
    chars
)
#from src.log import get_logger

#logger = get_logger(__name__)
chars_ = [char for char in chars]
chars_.append("-")
chars_ = np.array(chars_)
def load_trained_model():
    with open(cf.USE_JSON) as f:
        json_string = f.read()
    model = model_from_json(json_string)
    model.load_weights(cf.USE_MODEL)\

    return model


def load_test_samples():

    test_set = TextSequenceGenerator(
        cf.WORDS_T,
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=cf.DOWNSAMPLE_FACTOR,
        shuffle=False
    )
    #del data
    return test_set

#获取二值图
def get_binary_graph(img_g):
    img=cv2.adaptiveThreshold(img_g,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY_INV,31,20)
    kernel= np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = 255-img
    return img

def predict(img,model_p):
    #model = load_trained_model()

    #input_data = model.get_layer('the_input').output
    #y_pred = model.get_layer('softmax').output
    #model_p = Model(inputs=input_data, outputs=y_pred)

    #for root, dirs, files in os.walk(files_dir):
        #for file in files[:]:
            #print (os.path.join(root,file))
    #img_file = os.path.join(root,file)

    
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)

    #img = np.expand_dims(img, axis=0)
    #print(img.shape)
    net_out_value = model_p.predict(img)
    
    pred_texts = decode_predict_ctc(net_out_value, chars_)

    return pred_texts

if __name__ == '__main__':
    import random  # noqa
    model = load_trained_model()

    input_data = model.get_layer('the_input').output
    y_pred = model.get_layer('softmax').output
    model_p = Model(inputs=input_data, outputs=y_pred)
    all_num=0
    true_num=0
    false_num=0
            
    with open(cf.WORDS_TEST) as f:
        for line in f:
            line_split = line.strip().split(' ')
            #if line_split[0].split('-')[1] !="imgs":
                #print(line_split[0].split('-')[1])
            img_path=cf.WORDS_FOLDER+"/".join(line_split[0].split('-'))+".jpg"
            with open(cf.WORDS_TEST_JSON,'r') as load_f:
                load_dict = json.load(load_f)
            re_text = load_dict[line_split[0]]
            #if "BanQian" in img_path  and "train63" in img_path:
            img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #获取二值图
            try:
                img = get_binary_graph(img)
            except:
                print("F")
            img = preprocess(
                img,
                cf.IMAGE_SIZE, False
            )
            pre_text = predict(np.array([img]),model_p)[0]
            #pre_text = re.sub("\)|\(|[0-9]|=", "",pre_text)
            pre_text = re.sub("=", "",pre_text)
            re_text = re.sub("=", "",re_text)
            all_num+=1
            pre_text = pre_text#.lower()
            re_text = re_text#.lower().strip()
            if re_text==pre_text :#or re_text==pre_text[1:]
                true_num+=1
            else:
                print(img_path)
                print(re_text+" : "+pre_text)
                false_num+=1
                print(false_num,true_num,all_num,true_num/all_num)
            if all_num >1000:
                break
        #print(false_num,true_num,all_num,true_num/all_num)
