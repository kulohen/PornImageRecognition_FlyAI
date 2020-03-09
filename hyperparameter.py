import argparse
from processor import img_size

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=300, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()

num_classes = 5
# 训练集的每类的batch的量，组成的list
train_batch_List = [ 80 ] * num_classes
# 验证集的batch量，模拟预测集
val_batch_size = {
    0: 42,
    1: 40,
    2: 38,
    3: 41,
    4: 39
}
#TODO 合并并且重新分割train-set和val的比例

#是否开启每一类的验证,True代表开启（影响速率）
val_per_class = False

train_epoch = args.EPOCHS
train_batch = args.BATCH

# 保存最佳model的精准度，比如1%的准确范围写1,若2%的保存范围写2
save_boundary =1.5

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


#TODO  print参数，以便在flyai log里查看
def printHpperparameter():
    for s in param_list:
        print(s)
    pass


if __name__=='__main__':
    printHpperparameter()
    # print(param_list_dict)