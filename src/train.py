from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
import sys
sys.path.append("/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master")

import src.config as cf

from src.data_generator import TextSequenceGenerator
from src.models import CRNN_model
from keras.models import Model
#from src.log import get_logger

#logger = get_logger(__name__)
from keras.models import Model, model_from_json
def load_trained_model():
    with open(cf.USE_JSON) as f:
        json_string = f.read()
    # model = model_from_json(json_string)
    model.load_weights(cf.USE_MODEL_ADD)\

    return model

def train():

    train_set = TextSequenceGenerator(
        cf.WORDS_TRAIN,
        batch_size=cf.BATCH_SIZE,
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=cf.DOWNSAMPLE_FACTOR
    )
    test_set = TextSequenceGenerator(
        cf.WORDS_VAL,
        batch_size=cf.BATCH_SIZE,
        img_size=cf.IMAGE_SIZE, max_text_len=cf.MAX_LEN_TEXT,
        downsample_factor=cf.DOWNSAMPLE_FACTOR,
        shuffle=False, data_aug=False
    )

    no_train_set = max(train_set.ids)
    #print(no_train_set)
    no_val_set = max(test_set.ids)
    #logger.info("No train set: %d", no_train_set)
    #logger.info("No val set: %d", no_val_set)

    model, y_func = CRNN_model()
    

    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    model.load_weights(cf.USE_MODEL_ADD)
    ckp = ModelCheckpoint(
        cf.MODEL_CHECKPOINT, monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='min'
    )
    
    model.fit_generator(generator=train_set,
                        steps_per_epoch=no_train_set // cf.BATCH_SIZE,
                        epochs=cf.NO_EPOCHS,
                        validation_data=test_set,
                        validation_steps=no_val_set // cf.BATCH_SIZE,
                        callbacks=[ckp, earlystop])

    return model, y_func


if __name__ == '__main__':
    model, test_func = train()

    model_json = model.to_json()
    with open(cf.CONFIG_MODEL, 'w') as f:
        f.write(model_json)

    model.save_weights(cf.WEIGHT_MODEL)
