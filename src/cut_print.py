import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import predict_change

import sys
import json
sys.path.append("/home/season/Desktop/DK/CRNN_CTC_English_Handwriting_Recognition-master")
import src.config as cf
from keras.models import Model
def get_good_img(img,close_y,close_x,open_y,open_x):#
    
    name = time.time()
    # cv2.imwrite("duijie_img_debug/"+str(name)+'1.jpg', img)
    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY_INV,31,20)
    #
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    #cv2.imwrite("duijie_img_debug/"+str(name)+'c.jpg', img)
    kernel = np.ones((close_y,close_x),np.uint8)
    close= cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite("duijie_img_debug/"+str(name)+'d.jpg', close)
    kernel = np.ones((open_y,open_x),np.uint8)
    open_img = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)

    return open_img


def get_index(arr,th):
    top = np.where(arr>th)[0]
    index_dic = top[1:]-top[:-1]
    top_dic_index = np.where(index_dic>1)[0]
    part_right_index = top[top_dic_index]
    right_index = np.zeros(len(part_right_index)+1)
    right_index[-1] = len(arr)
    right_index[:-1] = part_right_index
    right_index = np.int64(right_index+1)

    part_left_index = top[top_dic_index]+index_dic[top_dic_index]
    left_index = np.zeros(len(part_left_index)+1)
    left_index[1:] = part_left_index
    left_index = np.int64(left_index)
    bottom_area = np.zeros(len(part_left_index))
    for num in range(len(part_left_index)):
        area = arr[part_right_index[num]:part_left_index[num]]
        if len(area)>0:
            min_index = np.where(area==np.min(area))[0]+part_right_index[num]
            bottom_area[num] = min_index[int(len(min_index)/2)]

    top_area = np.zeros(len(left_index))
    for num in range(len(left_index)):
        area = arr[left_index[num]:right_index[num]]
        if len(area)>0:
            max_index = np.where(area==np.max(area))[0]+left_index[num]
            top_area[num] = max_index[int(len(max_index)/2)]
    return top_area,bottom_area
def get_row(img):
    
    img_mean_y = img.mean(axis=1)
    top_area,bottom_area = get_index(img_mean_y,5)
    fir_row = np.where(img_mean_y>0)[0]
    if len(fir_row) >0:
        fir_row = np.where(img_mean_y>0)[0][0]
    else:
        fir_row=0
    end_row = np.where(img_mean_y>0)[0]
    if len(end_row) >0:
        end_row = np.where(img_mean_y>0)[0][-1]
    else:
        end_row=len(img_mean_y)-1
    if fir_row >10:
        fir_row-=10
    else:
        fir_row=0
    if len(img_mean_y)-1-end_row>10:
        end_row+=10
    else:
        end_row=len(img_mean_y)-1
    return bottom_area,fir_row,end_row
def get_column(img,first_row,second_row):
    img = get_good_img(img[int(first_row):int(second_row),:],30,12,6,6)
    y = img.mean(axis=0)
    top_area_x,bottom_area_x = get_index(y,5)
    fir_column = np.where(y>0)[0]
    if len(fir_column)>0:
        fir_column = np.where(y>0)[0][0]
    else:
        fir_column=0
    end_column = np.where(y>0)[0]
    if len(end_column)>0:
        end_column = np.where(y>0)[0][-1]
    else:
        end_column = len(y)-1
    if fir_column > 10:
        fir_column-=10
    else:
        fir_column=0
    if len(y)-1-end_column>10:
        end_column+=10
    else:
        end_column=len(y)-1
    return bottom_area_x,fir_column,end_column
def get_all_column(good_img,row):
    all_column_area = []
    for idx in range(len(row)-1):
        column_area,fir_column,end_column = get_column(good_img,row[idx],row[idx+1])
        new_column_area = np.zeros(len(column_area)+2)
        new_column_area[0]=fir_column
        new_column_area[1:-1]=column_area
        new_column_area[-1]=end_column
        all_column_area.append(new_column_area)
    return all_column_area
def run(img):
    good_img = get_good_img(img,5,30,6,6)
    row_area,fir_row,end_row = get_row(good_img)
    new_row_area = np.zeros(len(row_area)+2)
    new_row_area[0]=fir_row
    new_row_area[1:-1]=row_area+3
    new_row_area[-1]=end_row
    all_column_area = get_all_column(img,new_row_area)
    words_index = []
    for num in range(1,len(all_column_area)+1):
        word_column = np.zeros((len(all_column_area[num-1])-1,4))
        word_column[:,1] = new_row_area[num-1]
        word_column[:,3] = new_row_area[num]
        word_column[:,0] = all_column_area[num-1][:-1]
        word_column[:,2] = all_column_area[num-1][1:]
        words_index.append(word_column)
    return words_index
if __name__ == '__main__':

    model = predict_change.load_trained_model()

    input_data = model.get_layer('the_input').output
    y_pred = model.get_layer('softmax').output
    model_p = Model(inputs=input_data, outputs=y_pred)

    for i in range(1,9):
        img_path='../data/print_data/'+str(i)+'.jpg'
        img=cv2.imread(img_path,0)
        area = run(img)
        img_part = []
        for row in area:
            for col in row:
                an_img = img[int(col[1]):int(col[3]),int(col[0]):int(col[2])]
                
                #name = time.time()
                #cv2.imwrite("../data/print_data/debug/"+str(name)+'1.jpg', an_img)
                an_img = predict_change.preprocess(
                    an_img,
                    cf.IMAGE_SIZE, False
                )
                
                #txt = predict_change.predict(np.array([an_img]),model_p)
                #print(txt)
                img_part.append(an_img)
        img_part=np.array(img_part)
        txt = predict_change.predict(img_part,model_p)
        print(txt)
        #plt.imshow(img)
        #plt.show()
            
