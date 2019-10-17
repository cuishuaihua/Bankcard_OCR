# 1.系统环境

* ubuntu18.04

# 2.软件
* anaconda3.5.0
* pycharm
#3.第三方库
* cuda 8.0
* cudnn 7.1.3
* tensorflow-gpu 1.8.0
* keras 2.2.4
* numpy 1.14.0
* pillow 5.1.0
* opencv-python 3.4.2.17
#4.环境配置
```angular2
sh setup.sh
```
* 注：若是选择使用cpu加速则去掉cpu部分注释，若是选择使用gpu加速则去掉gpu部分注释
#5.Demo
将测试图片放置demo/test_images下，检测结果会保存在demo/test_result下，识别结果会保存在demo/test_result/result.txt里
```
python3 demo.py
```

###注：模型训练配置见项目文档