import argparse
from processor import img_size

'''
迭代版本 30
in-res-v2
最终评分 86
EPOCHS 100
BATCH SIZE 32
耗时114分钟
提交时间 2020-03-09 18-27-02

迭代版本 29
in-res-v2
最终评分 84.2
EPOCHS 100
BATCH SIZE 16
耗时106分钟
提交时间 2020-03-09 16-58-41

迭代版本 28
resnet50
EPOCHS 300 BATCH SIZE 32 提交时间 2020-03-09 08-52-48 
当前best_score_acc :0.8100【50epoch，和16,24目测一致】

迭代版本 27
resnet50
最终评分 82.33
EPOCHS 300 BATCH SIZE 24 耗时243分钟 提交时间 2020-03-09 08-52-21 

迭代版本 26
resnet50,200 on val
最终评分 82.8
EPOCHS 300 BATCH SIZE 16 耗时269分钟 提交时间 2020-03-09 08-47-58 

迭代版本 25
resnet50,
最终评分 81.67
EPOCHS 300 BATCH SIZE 16 耗时236分钟 提交时间 2020-03-09 01-11-33 

迭代版本 24
dense121
最终评分 77
EPOCHS 300 BATCH SIZE 16 耗时194分钟 提交时间 2020-03-09 00-50-57 

迭代版本 23
dense121
最终评分 79.07
EPOCHS 300 BATCH SIZE 24 耗时373分钟 提交时间 2020-03-08 00-35-24 
'''

num_classes = 5

# 一、构建网络
# net.py修改

# 二、数据结构

# 数据增强倍数
per_train_ratio = 3
# 保存最佳model的精准度，比如1%的准确范围写1,若2%的保存范围写2
save_boundary =1.5


#TODO 合并并且重新分割train-set和val的比例

# 三、成绩（调参）：

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=300, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()



train_epoch = args.EPOCHS
train_batch = args.BATCH


# 随机学习率启动per epoch
random_per_epoch = 20

# 多岁epochs后降低学习率
reduce_lr_per_epochs = 8


# 四、性能or速度
#是否开启每一类的验证,True代表开启（影响速率）
val_per_class = False

# 训练集的每类的batch的量，组成的list
train_batch_List = [ 80 ] * num_classes
# 验证集的batch量，模拟预测集
val_batch_size = {
    0: 84,
    1: 79,
    2: 76,
    3: 83,
    4: 78
}





param_list =[
    '【参数情况】'
    '一/构建网络',
    '框架/神经网络修改  ',
    '冻结训练层  ',
    'learn transfer  ',
    '激活函数linear line 非relu sigmoid',
    '',
    '二、数据结构',
    '训练数据平衡,%s'%'是',
    '图片分辨率,%d:%d'%(img_size[0],img_size[1]),
    '重置train:val的数据量比例,%.2f'%0.8,
    '数据增强  ,%s'%'是',
    '保存model的条件,%.2f'%save_boundary,
    'Train set dropout0.5（一定程度避开噪音，不一定奏效）  ,%s'%'是',
    '',
    '三、成绩（调参）：  ',
    'epoch 介于15-20  ，%d'%train_epoch,
    'learn rate  ',
    'batch影响梯度  ,%d'%train_batch,
    '',
    '四、性能or速度  ',
    '减少val_batch的量  ',
    '轻度框架  '
]


#  print参数，以便在flyai log里查看
def printHpperparameter():
    for s in param_list:
        print(s)
    pass


if __name__=='__main__':
    printHpperparameter()
    # print(param_list_dict)