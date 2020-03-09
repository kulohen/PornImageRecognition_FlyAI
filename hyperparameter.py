'''

一/成绩（调参）：
框架/神经网络修改
冻结训练层
learn transfer
激活函数linear line 非relu sigmoid
dropout
epoch 介于15-20
learn rate
batch影响梯度
图片分辨率
重置train:val的数据量比例
数据增强

二/速度
减少VAL的量
轻度框架
'''

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

train_epoch = 16
history_train = 0
history_test = 0
best_score_by_acc = 0.
best_score_by_loss = 999.
lr_level = 0

#TODO  print参数，以便在flyai log里查看