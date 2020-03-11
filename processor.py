# -*- coding: utf-8 -*
import cv2
import numpy
from flyai.processor.base import Base
from flyai.processor.download import check_download
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from path import DATA_PATH

'''
把样例项目中的processor.py件复制过来替换即可
'''
img_size = [352, 352]

class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        img = image.load_img(path, target_size=(img_size[0], img_size[1]))
        x_data = image.img_to_array(img)
        x_data = preprocess_input(x_data)
        return x_data


    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def output_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        img = image.load_img(path, target_size=(img_size[0], img_size[1]))
        x_data = image.img_to_array(img)
        x_data = preprocess_input(x_data)
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