📅2020/03/09  
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

292行 util里，drop 0.5还没验证

2020/03/06 initial
解决了DEV.CSV重复2次的bug
写个plot?

2020-3-7
OPENCV读取有问题，修改为keras自带的读取功能，不行就上PIL

预测集的分布比例  
4-19.53  
3-20.6  
2-19.07  
1-19.8  
0-21  
train:val 0.9:0.1 改到 0.8：0.2  
val-set 扩大到和验证集基本靠近  
for循环里train set 扩大3倍（加快运算时间，减少验证次数）  
考虑添加callback 降低学习率  

新增超参功能，对经常调参的数据作集中化归类