#-*- coding:utf-8 –*-
from keras import backend as K
import time

CHARS=' !"#&\'()*+,-./0123456789:;=>?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
#CHARS = " ABCDEFIJLMNOSTVWabcdefghijklmnopqrstuvwxyz"
#CHARS = "".join(['A', 'B', 'C', 'D', 'a', 'c', 'd', 'e', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y'])
#CHARS = "".join([' ', '\'', ',', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '’','·'])
NO_LABELS = len(CHARS)+1
# params
MAX_LEN_TEXT = 30
IMAGE_SIZE = (128, 32)
IMG_W, IMG_H = IMAGE_SIZE
NO_CHANNELS = 1

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (NO_CHANNELS, IMG_W, IMG_H)
else:
    INPUT_SHAPE = (IMG_W, IMG_H, NO_CHANNELS)

NO_EPOCHS = 100
BATCH_SIZE = 32
CONV_FILTERS = 16
KERNEL_SIZE = (3, 3)
POOL_SIZE = 2
DOWNSAMPLE_FACTOR = POOL_SIZE ** 2
TIME_DENSE_SIZE = 32
RNN_SIZE = 512

# paths
name = str(time.time())
WORDS_DATA = '../data/txt/self_words.txt'
#WORDS_TRAIN = '../data/add_train/add_train_25_1_10_self.txt'
#WORDS_VAL = '../data/add_train/add_val_25_1_10_self.txt'
WORDS_TRAIN = '../data/txt/self_train.txt'
WORDS_VAL = '../data/txt/self_val.txt'

#lianxice  val
WORDS_TEST = '../data/txt/self_val.txt'
WORDS_TEST_JSON = '../data/txt/self_words.json'

#网上找数字加字母test
#WORDS_TEST = '../data/add_train/data_test.txt'
#WORDS_TEST_JSON = "../data/add_train/data.json"

#自写单词test
#WORDS_TEST = "../data/train/txt/test_self.txt"
#WORDS_TEST_JSON="../data/train/txt/all_words.json"

#纯手写test
#WORDS_TEST = '../data/train/txt/write.txt'
#WORDS_TEST_JSON = "../data/train/txt/words.json"

#增加自写abcd  test
#WORDS_TEST = '../data/add_train/add_wolkard_test.txt'
#WORDS_TEST_JSON="../data/add_train/add_wolkard_test.json"

#自写单词 val
#WORDS_TEST = '../data/train/txt/val_self.txt'
#WORDS_TEST_JSON="../data/train/txt/all_words.json"

#自写单词+网上找数字加字母  val1:2
#WORDS_TEST = '../data/add_train/add_val_25_self.txt'
#WORDS_TEST_JSON="../data/train/txt/all_words.json"

#自写单词+网上找数字加字母  val 1:10
#WORDS_TEST = '../data/add_train/add_val_25_1_10_self.txt'
#WORDS_TEST_JSON="../data/train/txt/all_words.json"


CONFIG_MODEL = '../models/3_19_mark_'+name+'.json'
WEIGHT_MODEL = '../models/3_19_mark_'+name+'.h5'
MODEL_CHECKPOINT = '../checkpoint/3_19_mark_'+name+'.hdf5'

# naming
WORDS_FOLDER = "../data/"

"""
data
├── words
│   ├── a01
│   ├── a02
│   ├── a03
│   ├── a04
│   ├── a05
...
"""

#name="great_add_500_2018_12_2_15_51"

#name = "2018_12_4___500"
#name = "2019_1_26___add_1_2_500_25"
#name = "2019_1_27___add_1_2_500_50"
#name = 'add_data_1548572119.728272'
#name = "add_data_1548602631.552481"
#name = "add_data_1548640838.4239588"
#name = "add_data_1548641940.6873336"
#name = "2019_1_28_self_30"
#name = "2019_1_28_self_30_add_1548686064.5852542"  #25 *62新 新旧1:2
#name = "2019_1_28_self_30_add_1_10_1548752958.3513715"  #25 *62新 新旧1:10
#name="3_19_mark_1553015983.8060675"
name = "3_19_mark_1553176229.9929674"
USE_JSON="/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master/models/"+name+".json"
USE_MODEL="/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master/models/"+name+".h5"


#name="2019_1_28_self_30"
#USE_JSON_ADD="/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master/models/"+name+".json"
#USE_MODEL_ADD="/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master/models/"+name+".h5"
USE_MODEL_ADD="F"
