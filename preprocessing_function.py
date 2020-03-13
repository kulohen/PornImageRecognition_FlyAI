from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import misc
import numpy
import numpy.random
import scipy.ndimage
import scipy.misc

import sys
import os
import cv2
from processor import crop_image_center_80percent_to_input_function
from  PIL import Image

from keras.preprocessing import image
# 训练数据的路径
out_PATH = os.path.join(sys.path[0], 'wangyi_test', 'preprocess_images.jpg')
image_path = os.path.join(sys.path[0], 'data', 'input','images','134.jpg')

scale_num = 0.75

def random_crop_image(image):

      assert image.shape[2] == 3
      height, width = image.shape[0], image.shape[1]
      dy, dx = int (height * scale_num),int(width * scale_num)
      x = numpy.random.randint(0, width - dx + 1)
      y = numpy.random.randint(0, height- dy + 1)

      image_crop = image[y:dy+y ,   x:dx+x, 0:3]

      image_crop = cv2.resize(image_crop, (width,height))

      return image_crop

# datagen = ImageDataGenerator(
#
#         preprocessing_function=crop_image_center_80percent_to_input_function
# )
#
# img = load_img(image_path)
# x = img_to_array(img)
# x = x.reshape((1,) + x.shape)
#
# i = 0
# for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',save_prefix=out_PATH, save_format='jpg'):
#     i += 1
#     if i > 20:
#         break  # 否则生成器会退出循环

data = cv2.imread(image_path)
data = cv2.resize(data , (373,373))
data_1 = crop_image_center_80percent_to_input_function(data)
cv2.imshow('80% picture',data_1)
cv2.imwrite(out_PATH,data_1)
