# -*- coding: utf-8 -*
import cv2
import numpy
from flyai.processor.base import Base
from flyai.processor.download import check_download
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from path import DATA_PATH

# 输入train的size
img_size = [384, 384]

'''
以下是crop_image_center_80percent_to_input_function的静态变量
减少计算量
'''
# 保留的图片中心的比例，如80%，写0.8
center_scale = 0.8

# input_x的size，然后处理成img_size
# origin_size = [int (img_size[0] / 0.8) , int (img_size[1] / 0.8)]
#
# x = int(origin_size[1] * (1 - center_scale) / 2)
# y = int(origin_size[0] * (1 - center_scale) / 2)
# dx = x + img_size[1]
# dy = y + img_size[0]

origin_size = [480, 480]
x = 48
y = 48
dx = 432
dy = 432

'''
把样例项目中的processor.py件复制过来替换即可
'''
def crop_image_center_80percent_to_input_function(input_image):
    assert input_image.shape[2] == 3
    image_crop = input_image[y: dy , x: dx, 0:3]
    return image_crop

class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        img = image.load_img(path, target_size=(origin_size[0], origin_size[1]))
        x_data = image.img_to_array(img)
        x_data = preprocess_input(x_data)
        x_data = crop_image_center_80percent_to_input_function(x_data)
        return x_data


    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def output_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        img = image.load_img(path, target_size=(origin_size[0], origin_size[1]))
        x_data = image.img_to_array(img)
        x_data = preprocess_input(x_data)
        x_data = crop_image_center_80percent_to_input_function(x_data)
        return x_data

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_y(self, label):
        one_hot_label = numpy.zeros([5])  ##生成全0矩阵
        one_hot_label[label] = 1  ##相应标签位置置
        return one_hot_label

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        return numpy.argmax(data)

if __name__=='__main__':
    print(origin_size)
    img_size = [111,111]
    print(origin_size)
    print((x, y, dx, dy))