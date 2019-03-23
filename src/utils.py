#-*- coding:utf-8 –*-
import random

import cv2,time
import math,random,copy
import numpy as np
from keras import backend as K
import sys
sys.path.append("/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master")
import src.config as cf


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def preprocess(img, img_size ,path, data_aug=False):
    
    """
    Put img into target img of size img_size
    Transpose for TF and normalize gray-values
    """
    
    
    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        print("img None ! ",path)
        img = np.zeros([img_size[1], img_size[0]])
    else:
        #随机上下左右add 
        
        try:
            img = get_binary_graph(img)
            img = remove_and_add_side(img)
            #print("T ",path)
        except:
            print("F ",path)
            #print(img.shape)
    # increase dataset size by applying random stretches to the images
    if data_aug:
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)),
                         1)  # random width, but at least 1
        img = cv2.resize(
            img, (wStretched,
                  img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5
    #cv2.imwrite("../data/see/train/"+str(time.time())+".png",img)#path.split("/")[-1]
    # create target image and copy sample image into it
    (wt, ht) = img_size
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    new_size = (max(min(wt, int(w / f)), 1), max(
        min(ht, int(h / f)),
        1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, new_size)
    target = np.ones([ht, wt]) * 255
    target[0:new_size[1], 0:new_size[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img

#随机上下左右修改为近似背景
def add_background(img_g):
    
    img=cv2.adaptiveThreshold(img_g,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY_INV,31,20)

    kernel = np.ones((10,10),np.uint8) 
    img =cv2.dilate(img,kernel,iterations = 1)

    blackground_index_x = np.where(img<100)[1]
    blackground_index_y = np.where(img<100)[0]
    q_x = np.where(img>100)[1]
    q_y = np.where(img>100)[0]

    background = img_g[blackground_index_y,blackground_index_x]

    k=img_g.shape[0]/img_g.shape[1]
    w=math.sqrt(len(background)/k)
    h=k*w
    background = background[:int(w)*int(h)]
    background = background.reshape(int(h),int(w))

    left = random.randint(1,10)
    right = random.randint(1,10)
    top = random.randint(1,5)
    bottom = random.randint(1,5)
    
    new_img = cv2.resize(background,
                                (img_g.shape[1]+left+right,
                                img_g.shape[0]+top+bottom),
                                interpolation=cv2.INTER_CUBIC )
    new_q_x = np.array(q_x)+left
    new_q_y = np.array(q_y)+top
    new_img[new_q_y,new_q_x]=img_g[q_y,q_x]
    return new_img

#获取二值图
def get_binary_graph(img_g):
    img=cv2.adaptiveThreshold(img_g,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY_INV,51,20)
    kernel= np.ones((2,2),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = 255-img
    return img
#随机 remove side  and  add side
def remove_and_add_side(img_g):
    img_index = np.where(img_g<150)
    y = max(img_index[0])-min(img_index[0])
    x = max(img_index[1])-min(img_index[1])
    left = 1#random.randint(1,10)
    right = 1
    top = random.randint(1,5)
    bottom = random.randint(1,5)
    new_img = np.ones((y+top+bottom,x+left+right))*255
    new_img[top:-bottom,left:-right] = img_g[min(img_index[0]):max(img_index[0]),min(img_index[1]):max(img_index[1])]
    return new_img

def train_test_split_file(train_size=0.95):
    no_lines = 0
    with open(cf.WORDS_DATA) as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            no_lines += 1
    count = 0
    index_split = int(train_size * no_lines)

    with open(cf.WORDS_DATA) as f, \
            open(cf.WORDS_TRAIN, "w") as f_train, \
            open(cf.WORDS_TEXT, "w") as f_test:
        for line in f:
            if not line or line.startswith('#'):
                continue

            if count < index_split:
                f_train.write(line.strip() + "\n")
                count += 1
            else:
                f_test.write(line.strip() + "\n")
                count += 1


if __name__ == '__main__':
    train_test_split_file(train_size=0.95)

