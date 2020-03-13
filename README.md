2020-3-13  
input里resize到373* 373，再裁剪中间的299*299  
采用363 * 363  
学习率抛弃1e-3 和3e-4，从1e-4起步降低  

2020-3-12  
ResNeXt application 居然没有？  
https://stackoverflow.com/questions/57158756/unable-to-import-resnext-101-32%C3%9748d-from-keras-application  


2020-3-11     
NASnet large 费时费力，不太好用  
val 200 or 400，耗时几乎一样（155s差1秒）  
random—crop  
352*352 resnet50  

2020-3-10  
启用 363*363 分辨率 或者更大  
save model by best acc （其中的最小loss）  
200 on val 改称400  
修正bug ：per 8 epoch 下降lr  

2020/03/09  
100 on val 改到 200
开局lr=0.001改到1e-4

调参总结：

一、构建网络  
框架/神经网络修改  
冻结训练层  
learn transfer  
激活函数linear line 非relu sigmoid

二、数据结构  
训练数据平衡  
图片分辨率  
重置train:val的数据量比例  
数据增强  
保存model的条件:精确值，save loss or acc  
Train set dropout0.5（一定程度避开噪音，不一定奏效）  

三、成绩（调参）：  
epoch 介于15-20  
learn rate  
batch影响梯度  

四、性能or速度  
减少val_batch的量  
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